#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>

// helper to load kernel source from file
static std::string loadKernel(const std::string &path) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("cannot open kernel file");
    return std::string(std::istreambuf_iterator<char>(in), {});
}

int main() {
    try {
        // choose first platform & device
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw std::runtime_error("no OpenCL platform found");

        cl::Platform plat = platforms[0];
        std::vector<cl::Device> devices;
        plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.empty()) throw std::runtime_error("no OpenCL device found");

        cl::Device dev = devices[0];
        cl::Context context(dev);
        cl::CommandQueue queue(context, dev);

        // load kernel source (our main.clcpp file)
        std::string src = loadKernel("main.clcpp");
        cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length()));
        cl::Program program(context, sources);
        program.build({dev});

        // example: compute a^b for each pair in arrays
        std::vector<int> a = {2, 3, 5, 10};
        std::vector<int> b = {8, 4, 3, 2};
        size_t n = a.size();
        std::vector<int> result(n);

        cl::Buffer bufA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(int)*n, a.data());
        cl::Buffer bufB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(int)*n, b.data());
        cl::Buffer bufR(context, CL_MEM_WRITE_ONLY, sizeof(int)*n);

        cl::Kernel kernel(program, "powi");
        kernel.setArg(0, bufA);
        kernel.setArg(1, bufB);
        kernel.setArg(2, bufR);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n));
        queue.enqueueReadBuffer(bufR, CL_TRUE, 0, sizeof(int)*n, result.data());

        for (size_t i = 0; i < n; ++i)
            std::cout << a[i] << "^" << b[i] << " = " << result[i] << '\n';
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
