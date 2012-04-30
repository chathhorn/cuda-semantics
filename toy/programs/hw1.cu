#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <sys/time.h>

#define NO_EDGE_FND -1
#define LIST_END -1

#define CUDA_DEVICE 0
#define THREADS_PER_BLOCK 256

// A vertex from a directed graph.
typedef struct VERTEX_t {
      // Unique vertex id.
      int num;
      // First incoming edge.
      int ei;
      // First outgoing edge.
      int eo;
      // The address of the maximum vertex in the cycle this vertex is a part
      // of, zero otherwise.
      int cyc;
      // Used by restoreSpanningTree to hold the adjusted maximum incoming edge
      // weight.
      int max_adj;
      // Used by restoreSpanningTree as a flag to indicate this vertex has been
      // processed.
      int proc;
} VERTEX_t;

// An edge from a directed graph.
typedef struct EDGE_t {
      // Source vertex.
      int vo;
      // Sink vertex.
      int vi;
      // Edge weight.
      int w;
      // Used by restoreSpanningTree.
      int adj_w;
      // The next source edge.
      int next_o;
      // The next sink edge.
      int next_i;
      // Used by trimSpanningTree & restoreSpanningTree to indicate an incoming
      // edge doesn't have the maximum weight of all incoming edges and/or
      // isn't selected.
      int rmvd;
      // Used by restoreSpanningTree.
      int buried;
} EDGE_t;

// A directed graph.
typedef struct DIGRAPH_t {
      // The vertices.
      VERTEX_t *v;
      // The edges.
      EDGE_t *e;
      // The number of vertices.
      int num_v;
      // The number of edges.
      int num_e;
} DIGRAPH_t;

/* ADD_COMMENT
 * - Describe the high-level functionality provided by the function below,
 *   along with their parameters.
 * - Specify which functions run on the host and which on the device.
 */

// Checks the return value of a CUDA function for an error, in which case it
// prints a corresponding error message and then calls exit(1). Runs on the
// host.
static void cudaCheckError(cudaError_t ce);

// Finishes initialization of the parameter DIGRAPH_t by adding the parameter
// edge to the incoming/outgoing edge lists of its incident vertices. Runs on
// the host.
static void addEdge(DIGRAPH_t* d, int addr);

// Locates the incoming edge with the maximum weight on every vertex in the
// parameter vertex array, moves it to the front of the incoming edge list, and
// marks every other incoming edge as removed. Runs on the device.
__global__ static void trimSpanningTree(EDGE_t* e, VERTEX_t* v, int num_v);

// Finds cycles in the parameter vertex array, marking each vertex with the
// largest vertex address of the cycle in which it is contained. Runs on the
// device.
__global__ static void findCycles(EDGE_t* e, VERTEX_t* v, int num_v);

// Reweights edges based on the maximum edge weight for edges entering a cycle
// and picks new edges for the MST based on the new weight. Returns 1 if the
// parameter graph still contains at least one cycle, 0 otherwise. Runs on the
// host.
static int restoreSpanningTree(DIGRAPH_t* d);

// Check that the parameter digraph is in fact a spanning tree (not necessarily
// the maximum, though). Returns the total weight of all edges in the tree,
// zero if it isn't a spanning tree. Runs on the host.
static unsigned checkMST(DIGRAPH_t* mst);

// Writes the parameter spanning tree digraph to the parameter file. Runs on
// the host.
static void writeMST(DIGRAPH_t* mst, FILE* file);

inline void cudaCheckError(cudaError_t ce) {
      if (ce != cudaSuccess) {
            printf("Cuda error: %s\n\n", cudaGetErrorString(ce));
            exit(1);
      }
}

/*** Task 2 profiler stuff ***/
#define PROF(what, kind) prof_start(&prof, (kind)); (what); prof_stop(&prof, (kind));

enum prof_func {
      PROF_MAIN = 0,
      PROF_MEM_TRANS,
      PROF_CUDA_CHECK_ERROR,
      PROF_ADD_EDGE,
      PROF_TRIM_SPANNING_TREE,
      PROF_FIND_CYCLES,
      PROF_RESTORE_SPANNING_TREE,
      PROF_ENUM_SIZE,
};

char const* prof_titles[] = {
      "main()",
      "Mem. Transfers",
      "cudaCheckError()",
      "addEdge()",
      "trimSpanningTree()",
      "findCycles()",
      "restoreSpanningTree()",
};

struct prof_data {
      struct timeval time_started[PROF_ENUM_SIZE];
      unsigned total_invocations[PROF_ENUM_SIZE];
      struct timeval total_times[PROF_ENUM_SIZE];
};

// Initialize, start, stop, and print the profiler data. Run on the host.
static void prof_init(struct prof_data* prof);
static void prof_start(struct prof_data* prof, enum prof_func which_func);
static void prof_stop(struct prof_data* prof, enum prof_func which_func);
static void prof_print_stats(struct prof_data* prof);

// Returns time1 - time0. Runs on the host.
static struct timeval time_sub(struct timeval const* time1, struct timeval const* time0);

// Returns time1 + time0. Runs on the host.
static struct timeval time_add(struct timeval const* time1, struct timeval const* time0);

void prof_init(struct prof_data* prof) {
      memset(prof->total_invocations, 0, sizeof(prof->total_invocations));
      memset(prof->total_times, 0, sizeof(prof->total_times));
}

void prof_start(struct prof_data* prof, enum prof_func which_func) {
      gettimeofday(&prof->time_started[which_func], NULL);
      ++(prof->total_invocations[which_func]);
}

void prof_stop(struct prof_data* prof, enum prof_func which_func) {
      struct timeval time_stopped, time_elapsed;
      gettimeofday(&time_stopped, NULL);

      time_elapsed = time_sub(&time_stopped, &prof->time_started[which_func]);

      prof->total_times[which_func] = 
            time_add(&time_elapsed, &prof->total_times[which_func]);
}

void prof_print_stats(struct prof_data* prof) {
      unsigned i;
      struct timeval avg_time;

      printf("%22s\t%13s\t%13s\t%13s\n",
                  "Function",
                  "# Invocations",
                  "Total Time (s)",
                  "Avg. Time (s)");
      puts("---------------------------------------------------------------------");

      for (i = 0; i != PROF_ENUM_SIZE; ++i) {
            avg_time = prof->total_times[i];
            avg_time.tv_sec /= prof->total_invocations[i];
            avg_time.tv_usec /= prof->total_invocations[i];
            
            printf("%22s\t%13u\t%6u.%06u\t%6u.%06u\n", 
                        prof_titles[i],
                        prof->total_invocations[i],
                        prof->total_times[i].tv_sec,
                        prof->total_times[i].tv_usec, 
                        avg_time.tv_sec, 
                        avg_time.tv_usec);

      }
}

struct timeval time_sub(struct timeval const* time1, struct timeval const* time0) {
      struct timeval result;

      result.tv_sec = time1->tv_sec - time0->tv_sec;

      if (time1->tv_usec > time0->tv_usec) {
            result.tv_usec = time1->tv_usec - time0->tv_usec;
      } else { 
            --result.tv_sec;
            result.tv_usec = (time1->tv_usec + 1000000) - time0->tv_usec;
      }

      return result;
}

struct timeval time_add(struct timeval const* time1, struct timeval const* time0) {
      struct timeval result;

      result.tv_sec = time0->tv_sec + time1->tv_sec;
      result.tv_usec = time0->tv_usec + time1->tv_usec;

      if (result.tv_usec >= 1000000) {
            ++result.tv_sec;
            result.tv_usec -= 1000000;
      }

      return result;
}

/*** Task 1 spanning tree check & print ***/

unsigned checkMST(DIGRAPH_t* mst) {
      VERTEX_t* v = mst->v;
      EDGE_t* e = mst->e;
      unsigned i, nedges = 0, total_weight = 0;

      for (i = 0; i != mst->num_e; ++i) {
            if (!e[i].rmvd) {
                  // Have we seen this vertex before?
                  if (v[e[i].vi].cyc) return 0;

                  v[e[i].vi].cyc = 1;
                  total_weight += e[i].w;
                  ++nedges;
            }
      }

      // Now check the root -- it shouldn't have any incoming edges.
      if (v[0].cyc) return 0;

      // n-1 edges are required to span a graph of n vertices.
      if (nedges != (mst->num_v-1)) return 0;

      return total_weight;
}

void writeMST(DIGRAPH_t* mst, FILE* fout) {
      VERTEX_t* v = mst->v;
      EDGE_t* e = mst->e;
      unsigned i;
      int cur_edge;

      for (i = 0; i != mst->num_v; ++i) {
            cur_edge = v[i].eo;
            while (cur_edge != LIST_END) { 
                  if (!e[cur_edge].rmvd) {
                        fprintf(fout, "%i\t%i\t%i\t\n", 
                              i, e[cur_edge].w, e[cur_edge].vi);
                  }
                  cur_edge = e[cur_edge].next_o;
            }
      }
}

// Move the incoming edge with the greatest weight to the front of every
// vertex's incoming edge list and mark the other incoming edges as removed.
__global__ void trimSpanningTree(EDGE_t* e, VERTEX_t* v, int num_v) {
      int max, max_addr, last, last_max, next;
      int id = blockIdx.x*blockDim.x + threadIdx.x;

      // Check if vertex is in bounds and not the root
      if ((id < num_v) && (id != 0)) {
            max = 0;
            max_addr = -1;
            last = -1;
            last_max = -1;

            // Get head of the linked list
            next = v[id].ei;

            // While the tail is not found
            while (next != -1) {
                  // Check max and mark
                  if (e[next].w > max) {
                        max = e[next].w;
                        if (max_addr != -1) {
                              // Remove old max
                              e[max_addr].rmvd = 1;
                        }
                        max_addr = next;
                        last_max = last;
                  // If not max mark removed
                  } else {
                        e[next].rmvd = 1;
                  }
                  // Store last and get next
                  last = next;
                  next = e[next].next_i;

            }

            // If not already at the front of the list, move it there
            if (last_max != -1) {
                  next = e[max_addr].next_i;

                  e[last_max].next_i = next;
                  e[max_addr].next_i = v[id].ei;
                  v[id].ei = max_addr;
            }
      }
}

__global__ void findCycles(EDGE_t* e, VERTEX_t* v, int num_v) {
      unsigned i;

      int id = blockIdx.x*blockDim.x + threadIdx.x;

      int curr;
      int max = 0;

      // Check if vertex is in bounds and not the root
      if (id < num_v && id != 0) {
            curr = e[v[id].ei].vo;
            v[id].cyc = 0;

            // The edges can be backtracked (# of vertices) times until
            // it is known whether the initial vertex is connected to the root
            for (i = 0; i != num_v; ++i) {
                  // Check if root found
                  if (curr == 0) {
                        return;
                  }

                  // Keep track of the vertex with the greatest id.
                  if (curr > max) {
                        max = curr;
                  }

                  // We in a cycle.
                  if (curr == id) {
                        v[id].cyc = max;
                        return;
                  }

                  // Get next vertex
                  curr = e[v[curr].ei].vo;
            }
      }
}

void addEdge(DIGRAPH_t* d, int addr) {
      // Next list item
      int next;

      // Insert edge at head of the outgoing list
      next = d->v[d->e[addr].vo].eo;
      d->v[d->e[addr].vo].eo = addr;
      d->e[addr].next_o = next;

      // Insert edge at the head of the incoming list
      next = d->v[d->e[addr].vi].ei;
      d->v[d->e[addr].vi].ei = addr;
      d->e[addr].next_i = next;
}

int restoreSpanningTree(DIGRAPH_t* d) {
      int i = 0;

      int cyc_found = 0;

      int max = 0;

      int next, num_proc, cycle, cyc_max, cyc_max_addr, prev, after;

      // Find the max adjusted incoming for each vertex and store it in the vertex
      for (i = 0; i < d->num_v; i++) {
            if (d->v[i].cyc > 0) {
                  d->v[i].max_adj = NO_EDGE_FND;
                  max = 0;

                  next = d->v[i].ei;
                  while (next != LIST_END) {
                        if ((d->e[next].rmvd == 1) && (d->v[i].cyc != d->v[d->e[next].vo].cyc) && (d->e[next].buried != 1)) {
                              d->e[next].adj_w = d->e[next].w - d->e[d->v[i].ei].w;

                              if (d->e[next].w > max) {
                                    max = d->e[next].w;
                                    d->v[i].max_adj = next;
                              }
                        }
                        next = d->e[next].next_i;
                  }
            }

            d->v[i].proc = 0;
      }

      num_proc = 0;

      for (i = 0; i < d->num_v; i++) {
            if (d->v[i].cyc == 0) {
                  d->v[i].proc = 1;
                  num_proc++;
            } else {
                  cyc_found = 1;
            }
      }

      while (num_proc != d->num_v) {
            cycle = 0;
            cyc_max = -100000;
            cyc_max_addr = -1;

            for (i = 0; i < d->num_v; i++) {

                  if (d->v[i].proc == 1) {
                        continue;
                  }

                  if (cycle == 0) {
                        cycle = d->v[i].cyc;
                  }

                  if (d->v[i].cyc == cycle) {

                        if (d->v[i].max_adj != -1) {
                              if ((d->e[d->v[i].max_adj].adj_w > cyc_max)) {
                                    cyc_max = d->e[d->v[i].max_adj].adj_w;
                                    cyc_max_addr = d->v[i].max_adj;
                              }
                        }
                        d->v[i].proc = 1;

                        num_proc++;
                  }
            }

            d->e[d->v[d->e[cyc_max_addr].vi].ei].buried = 1;
            d->e[d->v[d->e[cyc_max_addr].vi].ei].rmvd = 1;
            d->e[cyc_max_addr].rmvd = 0;

            after = d->e[cyc_max_addr].next_i;

            next = d->v[d->e[cyc_max_addr].vi].ei;
            while (next != -1) {
                  if (d->e[next].next_i == cyc_max_addr) {
                        prev = next;
                  }
                  next = d->e[next].next_i;
            }

            d->e[prev].next_i = after;

            d->e[cyc_max_addr].next_i = d->v[d->e[cyc_max_addr].vi].ei;
            d->v[d->e[cyc_max_addr].vi].ei = cyc_max_addr;		
      }

      // First contraction is done, some information needs to be reinitialized

      for (i = 0; i < d->num_e; i++) {
            d->e[i].adj_w = -100000000;
      }

      for (i = 0; i < d->num_v; i++) {
            d->v[i].max_adj = -1;
      }

      return cyc_found;
}

int main(int argc, char** argv) {

      /* ADD_COMMENT
       * - Indicate the use of the variable below, and whether they reside on
       *   host or on device.
       */

      // Input/output file handles. Reside on the host.
      FILE* fin, *fout;

      // Counter and a flag indicating the digraph contains at least one cycle.
      // Both reside on the host.
      int i, fnd_c;

      // The digraph. Resides on the host.
      DIGRAPH_t d;

      // Points to the edge and vertex arrays. The *_gpu variants point to
      // device memory, the others to host memory.
      EDGE_t* e_gpu, *e;
      VERTEX_t* v_gpu, *v;

      // Required and available device memory.
      size_t dev_mem_req, dev_mem_ava;

      // CUDA device properties.
      struct cudaDeviceProp dev_props;

      // The total weight of our MST.
      unsigned total_weight = 0;

      // Number of iterations required to find our MST.
      unsigned niterations = 1;

      // Data for profiling.
      struct prof_data prof;

      // For capturing the return value of CUDA calls.
      cudaError_t cuda_err;

      prof_init(&prof);
      prof_start(&prof, PROF_MAIN);

      if (argc < 3) {
            printf("Error: Must have 2 argument" 
                   "(name of the file containing the directed graph and the output file\n");
            return -1;
      }

      fin = fopen(argv[1],"r");
      if (fin == NULL) {
            printf("Error: Could not open input file");
            return -1;
      }

      fscanf(fin,"%u",&d.num_v);
      fscanf(fin,"%u",&d.num_e);

      /* ADD_COMMENT
       * - instruction below 
       */
      // Set the device to be used for GPU executions and check the return
      // value for errors.
      cuda_err = cudaSetDevice(CUDA_DEVICE);

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      /* ADD_CODE
       * - compute the size (in Bytes) of the data structures that must be
       *   transferred to the device, and save it in dev_mem_req variable
       * - query the device memory availability of the GPU, and save it to
       *   dev_mem_ava
       * - print the hardware configuration of the underlying device
       * - if dev_mem_req > 75% of dev_mem_ava, stop and return the message:
       *   "too much memory required on device"
       */

      dev_mem_req = d.num_v*sizeof(VERTEX_t) + d.num_e*sizeof(EDGE_t);

      cuda_err = cudaGetDeviceProperties(&dev_props, CUDA_DEVICE);

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      dev_mem_ava = dev_props.totalGlobalMem;

      printf("*** CUDA device %u ***\n"
             "Name: %s\n"
             "Total global memory: %u\n"
             "Shared mem per block: %u\n"
             "Warp size: %d\n"
             "Mem pitch: %u\n"
             "Max threads per block: %d\n"
             "Compute capability: %u.%u\n"
             "\n", CUDA_DEVICE, dev_props.name, dev_props.totalGlobalMem,
             dev_props.sharedMemPerBlock, dev_props.warpSize,
             dev_props.memPitch, dev_props.maxThreadsPerBlock, dev_props.major,
             dev_props.minor);

      if (dev_mem_req > (0.75 * dev_mem_ava)) {
            puts("Too much memory required on device.");
            exit(1);
      }

      /* ADD_COMMENT
       * - instructions below (please add a comment at each blank line)
       */

      // Allocate host memory for vertices and edges.
      v = (VERTEX_t*) malloc(d.num_v * sizeof(VERTEX_t));
      e = (EDGE_t*) malloc(d.num_e * sizeof(EDGE_t));

      // Set the pointers in the digraph structure to the vertex and edge sets.
      d.v = v;
      d.e = e;

      // Zero the vertices and edges.
      memset(d.v, 0, sizeof(VERTEX_t) * d.num_v);
      memset(d.e, 0, sizeof(EDGE_t) * d.num_e);

      // Initialize the vertices with a unique id and invalid incoming and
      // outgoing edges.
      for (i = 0; i < d.num_v; i++) {
            d.v[i].num = i;
            d.v[i].eo = -1;
            d.v[i].ei = -1;
      }

      // Read the edges from the input file.
      for (i = 0; i < d.num_e; i++) {
            fscanf(fin, "%i\t%i\t%i", &d.e[i].vo, &d.e[i].vi, &d.e[i].w);
            PROF(addEdge(&d, i), PROF_ADD_EDGE);
      }

      // Initialize parameters for the CUDA kernel functions (number of threads
      // and blocks). We'll have approximately one thread per vertex.
      dim3 threads_per_block(THREADS_PER_BLOCK);
      dim3 blocks_per_grid(d.num_v / THREADS_PER_BLOCK 
                           + !!(d.num_v % THREADS_PER_BLOCK));

      // Allocate device memory to store the edge and vertex arrays (and check
      // cudaMalloc's return value for errors).
      cuda_err = cudaMalloc((void**) &e_gpu, d.num_e*sizeof(EDGE_t));

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      cuda_err = cudaMalloc((void**) &v_gpu, d.num_v*sizeof(VERTEX_t));

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      // Copy the vertex and edge arrays from host to device memory.
      PROF(cuda_err = cudaMemcpy((void*) e_gpu, d.e, d.num_e * sizeof(EDGE_t),
            cudaMemcpyHostToDevice), PROF_MEM_TRANS);

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      PROF(cuda_err = cudaMemcpy((void*) v_gpu, d.v, d.num_v *
           sizeof(VERTEX_t), cudaMemcpyHostToDevice), PROF_MEM_TRANS);

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      // Finds the incoming edge on each vertex with the max weight and flags
      // the other incoming edges as removed.
      prof_start(&prof, PROF_TRIM_SPANNING_TREE);
      trimSpanningTree<<<blocks_per_grid, threads_per_block>>>(
            e_gpu, v_gpu, d.num_v);
      cudaThreadSynchronize();
      prof_stop(&prof, PROF_TRIM_SPANNING_TREE);

      cuda_err = cudaGetLastError();

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      // Unroll the loop a bit.
      ++niterations;
      
      // Find and mark cycles in the current spanning tree.
      prof_start(&prof, PROF_FIND_CYCLES);
      findCycles<<<blocks_per_grid, threads_per_block>>>(
                  e_gpu, v_gpu, d.num_v);
      cudaThreadSynchronize();
      prof_stop(&prof, PROF_FIND_CYCLES);
      cuda_err = cudaGetLastError();
      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      // Copy the vertex and edge arrays back to the host from the device.
      PROF(cuda_err = cudaMemcpy((void*)d.e, (void*)e_gpu, d.num_e *
           sizeof(EDGE_t), cudaMemcpyDeviceToHost), PROF_MEM_TRANS);
      
      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      PROF(cuda_err = cudaMemcpy((void*)d.v, (void*)v_gpu, d.num_v *
           sizeof(VERTEX_t), cudaMemcpyDeviceToHost), PROF_MEM_TRANS);
      
      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      // Don't assume.
      PROF(fnd_c = restoreSpanningTree(&d), PROF_RESTORE_SPANNING_TREE);

      // Check for cycles and reweight and reform the MST until there are no
      // more cycles.
      while (fnd_c > 0) {
            // Increment the iterations counter so we know how many iterations
            // the process took.
            ++niterations;

            // Copy the vertex and edge arrays from host to device memory.
            PROF(cuda_err = cudaMemcpy((void*)e_gpu, d.e, d.num_e *
                 sizeof(EDGE_t), cudaMemcpyHostToDevice), PROF_MEM_TRANS);
      
            PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

            PROF(cuda_err = cudaMemcpy((void*) v_gpu, d.v, d.num_v *
                 sizeof(VERTEX_t), cudaMemcpyHostToDevice), PROF_MEM_TRANS);
      
            PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

            // Find and mark cycles in the current spanning tree.
            prof_start(&prof, PROF_FIND_CYCLES);
            findCycles<<<blocks_per_grid, threads_per_block>>>(
                        e_gpu, v_gpu, d.num_v);
            cudaThreadSynchronize();
            prof_stop(&prof, PROF_FIND_CYCLES);

            cuda_err = cudaGetLastError();

            PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

            // Copy the vertex and edge arrays back to the host from the
            // device.
            PROF(cuda_err = cudaMemcpy((void*)d.e, (void*)e_gpu, d.num_e *
                 sizeof(EDGE_t), cudaMemcpyDeviceToHost), PROF_MEM_TRANS);
           
            PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

            PROF(cuda_err = cudaMemcpy((void*)d.v, (void*)v_gpu, d.num_v *
                 sizeof(VERTEX_t), cudaMemcpyDeviceToHost), PROF_MEM_TRANS);

            PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

            // Reweight incoming edges based on the maximum incoming edge to
            // each cycle and then reform the MST based on the new weights.
            PROF(fnd_c = restoreSpanningTree(&d), PROF_RESTORE_SPANNING_TREE);
      }

      /* ADD_CODE
       * - Check whether the found MST is indeed a directed spanning tree. You
       *   can implement this in a separate function and invoke it here.
       * - Print the found MST into file (the file name should be the last
       *   argument to the program). You can implement this in a separate
       *   function and invoke it here.
       * - Print to stdout the weight of the MST, and the number of iterations
       *   needed to find it.
       */

      total_weight = checkMST(&d);

      if (!total_weight) {
            puts("We failed to find a maximum spanning tree.");
      } else {
            printf("We found a maximum spanning tree.\n"
                   "Total weight: %u\nTotal iterations: %u\n", 
                   total_weight, niterations);
      }

      fout = fopen(argv[2],"w");
      if (fout == NULL) {
            printf("Error: Could not open output file");
            return -1;
      }

      writeMST(&d, fout);

      free(e);
      free(v);

      cuda_err = cudaFree(e_gpu);

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      cuda_err = cudaFree(v_gpu);

      PROF(cudaCheckError(cuda_err), PROF_CUDA_CHECK_ERROR);

      prof_stop(&prof, PROF_MAIN);

      prof_print_stats(&prof);

      return 0;
}
