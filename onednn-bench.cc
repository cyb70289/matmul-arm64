// XXX: ACL doesn't support blas sgemm interface, oneDNN API dnnl_sgemm
//      won't trip ACL. This complex code enables ACL path in oneDNN.

#include <dnnl.hpp>

#include <chrono>
#include <iostream>

using namespace dnnl;

// Function to initialize a memory object with random data
void init_data(memory& mem) {
  auto dim = mem.get_desc().get_dims();
  size_t total_size = 1;
  for (auto& d : dim) total_size *= d;

  // Use float pointer to access the data
  float* p_data = (float*)mem.get_data_handle();
  for (size_t i = 0; i < total_size; ++i) {
    p_data[i] = i;
  }
}

// args: batch, m, n, k
int main(int argc, char* argv[]) {
  engine eng(engine::kind::cpu, 0);
  stream strm(eng);

  // default batch size and matrix shape
  int batch = 512;
  int m = 1000, n = 240, k = 200;
  if (argc == 5) {
    const int my_batch = std::atoi(argv[1]);
    const int my_m = std::atoi(argv[2]);
    const int my_n = std::atoi(argv[3]);
    const int my_k = std::atoi(argv[4]);
    if (my_batch <= 0 || my_m <= 0 || my_n <= 0 || my_k <= 0) {
      std::cerr << "invalid size\n";
      return 1;
    }
    batch = my_batch; m = my_m; n = my_n; k = my_k;
  }
  std::cout << "batch=" << batch << ", m=" << m << ", n=" << n << ", k=" << k << '\n';

  memory::desc a_md({batch, m, k}, memory::data_type::f32, memory::format_tag::abc);
  memory::desc b_md({batch, k, n}, memory::data_type::f32, memory::format_tag::abc);
  memory::desc c_md({batch, m, n}, memory::data_type::f32, memory::format_tag::abc);

  memory a_mem(a_md, eng);
  memory b_mem(b_md, eng);
  memory c_mem(c_md, eng);

  init_data(a_mem);
  init_data(b_mem);

  auto matmul_op = matmul::primitive_desc(eng, a_md, b_md, c_md);
  auto matmul_prim = matmul(matmul_op);

  // warmup
  {
    matmul_prim.execute(strm, {
        {DNNL_ARG_SRC, a_mem},
        {DNNL_ARG_WEIGHTS, b_mem},
        {DNNL_ARG_DST, c_mem}
        });
    strm.wait();
  }

  const auto start = std::chrono::high_resolution_clock::now();
  matmul_prim.execute(strm, {
      {DNNL_ARG_SRC, a_mem},
      {DNNL_ARG_WEIGHTS, b_mem},
      {DNNL_ARG_DST, c_mem}
      });
  strm.wait();
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "time: " << duration.count() << " ms\n";

  if (argc == 1) {
    // print some results for quick debugging
    float* c = (float*)c_mem.get_data_handle();
    std::cerr << "c[0]    = " << c[0] << '\n';
    std::cerr << "c[9973] = " << c[9973] << '\n';
    std::cerr << "c[-1]   = " << c[batch*m*n - 1] << '\n';
  }

  return 0;
}
