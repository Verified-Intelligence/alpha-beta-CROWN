/*
  Compile with "make"
*/

#include <cplexx.h>
#include <stdlib.h>
#include <stdint.h>
#include <string> // for string class
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>

using namespace std;

/* Declarations for functions in this program */

static int CPXPUBLIC
   cutCallbackShareRootCuts (CPXCENVptr env, void *cbdata, int wherefrom,
    void *cbhandle, int *useraction_p);

static void
   free_and_null (char **ptr),
   usage         (char *progname);

typedef struct {
    int call_count;
    int initial_rows;
    string filename;
    CPXDIM* row_indices;
    double* row_values;
    CPXNNZ row_storage_size;
} UserCallBackData;

typedef struct {
    int32_t signature = 0x53545543;  // "CUTS"
    int32_t num_rows;
    int32_t num_elements;
    int32_t row_begin_idx_offset;  // length is num_rows;
    int32_t rhs_values_offset;  // length is num_rows;
    int32_t row_indices_offset;  // length is num_elements;
    int32_t row_values_offset;  // length is num_elements;
} CutFileHeader;

typedef struct {
    int32_t signature = 0x58444e49;  // "INDX"
    int32_t first_col_num;  // first column index, usually 0.
    int32_t num_cols;
    int32_t names_offset;  // num_cols NULL terminated strings.
} IndexFileHeader;

// Pointer to CPXXgetrowname and CPXXgetcolname.
typedef int (*GetNamesFunc)(CPXCENVptr, CPXCLPptr, char**, char*, CPXSIZE, CPXSIZE*, CPXDIM, CPXDIM);


/* Get the name of rows in LP, from row begin to row end. */
void getNames (GetNamesFunc func, CPXCENVptr env, CPXCLPptr lp, CPXDIM begin, CPXDIM end, const char* filename=NULL, bool print=true) {
    CPXDIM length = end - begin + 1;
    size_t preallocated_size = length * 32;  // Assume each name is at most 20 characters.
    char** name = static_cast<char**>(malloc(length * sizeof(char*)));
    char* namestore = static_cast<char*>(malloc(preallocated_size * sizeof(char)));
    CPXSIZE surplus;
    int status;
    status = func(env, lp, name, namestore, preallocated_size, &surplus, begin, end);
    if (status) {
        fprintf(stderr, "CPXXgetrowname returns %d\n", status);
    }
    else {
        /* print all the names. */
        if (surplus < 0) {
            fprintf(stderr, "CPXXgetrowname does not have sufficient space, surplus=%lld\n", surplus);
        }
        else {
           CPXSIZE namestore_length = preallocated_size - surplus;
            if (print) {
                for (int i = 0; i < length; ++i) {
                    const char* s = name[i];
                    printf("idx %i name %s\n", i + begin, s);
                }
            }
            if (filename != NULL) {
                /* we implement exclusive writing by setting 000 permission when the file is creating,
                and change the permission to 444 when the file creation is finished */

                // first, allow us to modify the old file and then remove the old file if exists
                chmod(filename, S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH);
                if (access(filename, F_OK) != -1) {
                    remove(filename);
                }
                // create new file with 000 permission
                int fd = open(filename, O_CREAT | O_EXCL | O_WRONLY);
                if (fd != -1) {
                    try {
                        IndexFileHeader header;
                        header.first_col_num = begin;
                        header.num_cols = length;
                        header.names_offset = sizeof(IndexFileHeader);
                        printf("index header: begin %i num_cols %i names_offset %i \n", header.first_col_num, header.num_cols, header.names_offset);
                        // exit(0);
                        write(fd, &header, sizeof(header));
                        write(fd, namestore, sizeof(char) * namestore_length);
                        close(fd);
                        printf("%d names for indices saved to %s\n", length, filename);
                    } catch (int e) {
                        close(fd);
                        chmod(filename, S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH);
                        throw e;
                    }
                    close(fd);
                    chmod(filename, S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH);
                }
            }
        }
    }
    free(name);
    free(namestore);
}

void getRows (CPXCENVptr env, CPXCLPptr lp, CPXDIM begin, CPXDIM end, UserCallBackData* userdata, const char* filename=NULL, bool print=true) {
    CPXDIM length = end - begin + 1;
    CPXNNZ* row_begin_idx = static_cast<CPXNNZ*>(malloc(length * sizeof(CPXNNZ)));
    CPXNNZ nnz_count;
    CPXNNZ surplus;
    int status;
    printf("Extracting rows %d to %d\n", begin, end);
    do {
        // Get variable indices and their corresponding values in cuts.
        status = CPXXgetrows(env, lp, &nnz_count, row_begin_idx, userdata->row_indices, userdata->row_values,
              userdata->row_storage_size, &surplus, begin, end);
        if (status != 0 && status != CPXERR_NEGATIVE_SURPLUS) {
            fprintf(stderr, "CPXXgetrowname returns %d\n", status);
            break;
        }
        else if (surplus < 0) {
                // Space not enough to save all indices and values. Need to increase memory and call again.
                CPXNNZ new_storage_size = (userdata->row_storage_size - surplus) * 2;
                printf("CPXXgetrowname does not have sufficient space, surplus=%lld, increased size to %lld\n", surplus, new_storage_size);
                userdata->row_storage_size = new_storage_size;
                free(userdata->row_indices);
                free(userdata->row_values);
                userdata->row_indices = static_cast<CPXDIM*>(malloc(new_storage_size * sizeof(CPXDIM)));
                userdata->row_values = static_cast<double*>(malloc(new_storage_size * sizeof(double)));
        }
        else {
            // Now also get the rhs values.
            double* row_rhs = static_cast<double*>(malloc(length * sizeof(char*)));
            status = CPXXgetrhs(env, lp, row_rhs, begin, end);
            if (status) {
                fprintf(stderr, "CPXXgetrhs returns %d\n", status);
                break;
            }
            if (print) {
               for (int i = 0; i < length; ++i) {
                   CPXNNZ row_idx = row_begin_idx[i];
                   CPXNNZ row_end;  // Last element + 1 in this row.
                   if (i == length - 1) {
                       row_end = nnz_count;
                   }
                   else {
                       row_end = row_begin_idx[i+1];
                   }
                   printf("row %i: ", i + begin);
                   while (row_idx != row_end) {
                       printf("%8f * var_%04d ", userdata->row_values[row_idx], userdata->row_indices[row_idx]);
                       ++row_idx;
                   }
                   printf("<= %8f\n", row_rhs[row_idx]);
               }
            }
            if (filename != NULL) {
                /* we implement exclusive writing by setting 000 permission when the file is creating,
                and change the permission to 444 when the file creation is finished */

                // first, allow us to modify the old file and then remove the old file if exists
                chmod(filename, S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH);
                if (access(filename, F_OK) != -1) {
                    remove(filename);
                }
                // create new file with 000 permission
                int fd = open(filename, O_CREAT | O_EXCL | O_WRONLY);
                if (fd != -1) {
                    try {
                        CutFileHeader header;
                        header.num_rows = length;
                        header.num_elements = nnz_count;
                        header.row_begin_idx_offset = sizeof(CutFileHeader);
                        header.rhs_values_offset = header.row_begin_idx_offset + length * sizeof(CPXNNZ);
                        header.row_indices_offset = header.rhs_values_offset + length * sizeof(double);
                        header.row_values_offset = header.row_indices_offset + nnz_count * sizeof(CPXDIM);
                        write(fd, &header, sizeof(header));
                        write(fd, row_begin_idx, sizeof(CPXNNZ) * length);
                        write(fd, row_rhs, sizeof(double) * length);
                        write(fd, userdata->row_indices, sizeof(CPXDIM) * nnz_count);
                        write(fd, userdata->row_values, sizeof(double) * nnz_count);
                    } catch (int e) {
                        close(fd);
                        chmod(filename, S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH);
                        throw e;
                    }
                    close(fd);
                    chmod(filename, S_IRUSR|S_IRGRP|S_IROTH|S_IWUSR|S_IWGRP|S_IWOTH);
                }

                printf("%d rows saved to %s with %lld nonzeros\n", length, filename, nnz_count);
            }
            free(row_rhs);
        }
    } while(surplus < 0);
    free(row_begin_idx);
}

int cutCallbackShareRootCuts (CPXCENVptr env,
        void       *cbdata,
        int        wherefrom,
        void       *cbhandle,
        int        *useraction_p)
{
    int status;
    *useraction_p = CPX_CALLBACK_DEFAULT;
    UserCallBackData* userdata = static_cast<UserCallBackData*>(cbhandle);
    userdata->call_count++;

    /* Please be aware that call backs may be called from MULTIPLE THREADS!! May need locks to serialize the output procedure. */

    // CPX_CALLBACK_MIP_CUT_LAST indicates that CPLEX is done adding cuts and the user has a last chance to add cuts
    if(wherefrom == CPX_CALLBACK_MIP_CUT_LAST || wherefrom == CPX_CALLBACK_MIP_CUT_LOOP)
    {
        // With this, we stop the optimization after the callback is executed
        // *useraction_p = CPX_CALLBACK_FAIL;

        CPXLPptr tmpLp, copy;
        status = CPXXgetcallbacknodelp(env, cbdata, wherefrom, &tmpLp);
        copy = CPXXcloneprob(env, tmpLp, &status);  // TODO: this is probably not necessary.
        if ( status ) {
          fprintf (stderr, "Failed to clone LP.\n");
        }

        CPXDIM precols;
        CPXDIM prerows;
        char *prectype;
        precols = CPXXgetnumcols (env, copy);
        prerows = CPXXgetnumrows (env, copy);
        printf("Current problem size: %d rows, %d columns\n", prerows, precols);

        // Save the number of rows in initial problem without any cuts.
        // TODO: Verify we never got any cuts at the first time this callback is called.
        if (userdata->call_count == 1) {
            userdata->initial_rows = prerows;
            // Save all the column (variable) names to a .indx file.
            getNames(CPXXgetcolname, env, copy, 0, precols - 1, (userdata->filename + string(".indx")).c_str(), false);
        }
        else {
            printf ("Saving new cuts...\n");
            // getNames(CPXXgetrowname, env, copy, userdata->initial_rows, prerows - 1);
            // getNames(CPXXgetcolname, env, copy, 0, precols - 1);
            // Save all the cuts.
            getRows(env, copy, userdata->initial_rows, prerows - 1, userdata, (userdata->filename + string(".cuts")).c_str(), false);
             /*
            // Save mps file with cuts (TODO: remove after checking the cuts are corrected saved).
            prectype = static_cast<char*>(malloc (precols * sizeof(char)));
            status = CPXXgetcallbackctype (env, cbdata, wherefrom, prectype, 0, precols-1);
            status = CPXXcopyctype (env, copy, prectype);
            status = CPXXwriteprob(env, copy, (userdata->filename + string("-1stcuts.mps")).c_str(), NULL);
            if ( status ) {
                 fprintf (stderr, "Failed to write presolve/cut MIP.\n");
            }
            else {
                 printf ("Cuts saved to %s-1stcuts.mps\n", userdata->filename.c_str());
            }
            free(prectype);
            // stop at the first round cuts generation!!!
             exit(0);
             */
        }
    }

    return 0;

}

int
main (int  argc,
      char *argv[])
{
   int status = 0;

   if (argc < 3) {
      usage(argv[0]);
      return 1;
   }


   /* Declare and allocate space for the variables and arrays where
      we will store the optimization results, including the status,
      objective value, and variable values */

   int    solstat;
   string filename(argv[2]);

   CPXENVptr env = NULL;
   CPXLPptr  mip = NULL;
   /* Initialize the CPLEX environment */

   env = CPXXopenCPLEX (&status);

   /* Setup datastructure to be used in callback function. */
   UserCallBackData callback_data;
   callback_data.call_count = 0;
   callback_data.filename = filename;
   callback_data.row_storage_size = 1024;  // This will be dynamically increased if not enough.
   callback_data.row_indices = static_cast<CPXDIM*>(malloc(callback_data.row_storage_size * sizeof(CPXDIM)));
   callback_data.row_values = static_cast<double*>(malloc(callback_data.row_storage_size * sizeof(double)));

   /* If an error occurs, the status value indicates the reason for
      failure.  A call to CPXXgeterrorstring will produce the text of
      the error message.  Note that CPXXopenCPLEX produces no
      output, so the only way to see the cause of the error is to use
      CPXXgeterrorstring.  For other CPLEX routines, the errors will
      be seen if the CPXPARAM_ScreenOutput parameter is set to CPX_ON */

   if ( env == NULL ) {
      char errmsg[CPXMESSAGEBUFSIZE];
      fprintf (stderr, "Could not open CPLEX environment.\n");
      CPXXgeterrorstring (env, status, errmsg);
      fprintf (stderr, "%s", errmsg);
      goto TERMINATE;
   }

   /* Turn on output to the screen */

   status = CPXXsetintparam (env, CPXPARAM_ScreenOutput, CPX_ON);
   if ( status ) {
      fprintf (stderr,
               "Failure to turn on screen indicator, error %d.\n",
               status);
      goto TERMINATE;
   }

   /* Create the problem, using the filename as the problem name */

   mip = CPXXcreateprob (env, &status, argv[1]);

   /* A returned pointer of NULL may mean that not enough memory
      was available or there was some other problem.  In the case of
      failure, an error message will have been written to the error
      channel from inside CPLEX.  In this example, the setting of
      the parameter CPXPARAM_ScreenOutput causes the error message to
      appear on stdout.  Note that most CPLEX routines return
      an error code to indicate the reason for failure */

   if ( mip == NULL ) {
      fprintf (stderr, "Failed to create LP.\n");
      goto TERMINATE;
   }

   /* Now read the file, and copy the data into the created lp */

   status = CPXXreadcopyprob (env, mip, argv[1], NULL);
   if ( status ) {
      fprintf (stderr,
               "Failed to read and copy the problem data.\n");
      goto TERMINATE;
   }

   /* Set up to use MIP callbacks to get cuts. */

   status = CPXXsetusercutcallbackfunc(env, cutCallbackShareRootCuts, &callback_data);

   /* Set MIP log interval to 1 */

   status = CPXXsetcntparam (env, CPXPARAM_MIP_Interval, 1);
   if ( status )  goto TERMINATE;

   /* Set node limit to 1, so there is no branch and bound. */

   status = CPXXsetcntparam (env, CPXPARAM_MIP_Limits_Nodes, 1);
   if ( status )  goto TERMINATE;


   status = CPXXsetcntparam (env, CPXPARAM_Emphasis_MIP, CPX_MIPEMPHASIS_BESTBOUND);
   if ( status )  goto TERMINATE;

   /* Set to maximum cut number */

   status = CPXXsetcntparam (env, CPX_PARAM_CUTPASS, 2147483647);
   if ( status )  goto TERMINATE;

   /* Use multiple threads, but the callback may be called by multiple threads at the same time!! */

   status = CPXXsetcntparam (env, CPXPARAM_Threads, 10);
   if ( status )  goto TERMINATE;

   /* Allow non-deterministic result. Disable it for reproducibility.*/

   status = CPXXsetcntparam (env, CPXPARAM_Parallel, CPX_PARALLEL_OPPORTUNISTIC);
   if ( status )  goto TERMINATE;

   // TODO: warm-start the solver with known best adversarial candidates.

   /* Optimize the problem and obtain solution */

   status = CPXXmipopt (env, mip);
   if ( status ) {
      fprintf (stderr, "Failed to optimize MIP.\n");
      goto TERMINATE;
   }

   solstat = CPXXgetstat (env, mip);
   printf ("Solution status %d.\n", solstat);


TERMINATE:

   /* Free the problem as allocated by CPXXcreateprob and
      CPXXreadcopyprob, if necessary */

   if ( mip != NULL ) {
      int xstatus = CPXXfreeprob (env, &mip);

      if ( !status ) status = xstatus;
   }

   /* Free the CPLEX environment, if necessary */

   if ( env != NULL ) {
      int xstatus = CPXXcloseCPLEX (&env);

      if ( !status ) status = xstatus;
   }

   free(callback_data.row_indices);
   free(callback_data.row_values);

   return (status);

} /* END main */


/* This simple routine frees up the pointer *ptr, and sets *ptr to
   NULL */

static void
free_and_null (char **ptr)
{
   if ( *ptr != NULL ) {
      free (*ptr);
      *ptr = NULL;
   }
} /* END free_and_null */


static void
usage (char *progname)
{
   fprintf (stderr,
    "Usage: %s input_filename output_filename\n", progname);
   fprintf (stderr,
    "  input_filename, output_filename   Name of a file, with .mps, .lp, or .sav\n");
   fprintf (stderr,
    "             extension, and a possible, additional .gz\n");
   fprintf (stderr,
    "             extension\n");
} /* END usage */
