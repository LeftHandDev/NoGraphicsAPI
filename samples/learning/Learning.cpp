#include "stb_image.h"
#include "stb_image_write.h"

#include "Learning.h"
#include "Tensor.h"

#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <filesystem>
std::vector<float> load(std::string path)
{
    std::vector<float> res;
    int w, h, c;
    auto ptr = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!ptr)
    {
        throw std::runtime_error(std::string("failed to load image: ") + stbi_failure_reason());
    }
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            res.push_back(ptr[idx + 0] / 255.0f);
            res.push_back(ptr[idx + 1] / 255.0f);
            res.push_back(ptr[idx + 2] / 255.0f);
        }
    }
    stbi_image_free(ptr);
    return res;
}

void learningSample()
{
    Instance instance;
    auto device = instance.device(1);

    // try
    // {
    std::vector<float> gt = load("assets/Default.png");

    const unsigned int N = 256 * 256 * 3;

    class Autoencoder : public Module
    {
    public:
        Autoencoder(Device* device)
            : _device(device),
              enc1(device, 3, 8, 3, false, Tensor::Pad::reflect),
              enc2(device, 8, 16, 3, false, Tensor::Pad::reflect),
              enc3(device, 16, 32, 3, false, Tensor::Pad::reflect), // bottleneck: 32 channels
              dec1(device, 64, 16, 3, false, Tensor::Pad::reflect), // in = 32 (up) + 32 (skip s3)
              dec2(device, 32, 8, 3, false, Tensor::Pad::reflect),  // in = 16 (up) + 16 (skip s2)
              dec3(device, 16, 3, 3, false, Tensor::Pad::reflect),  // in =  8 (up) +  8 (skip s1)
              pool(2)
        {
        }

        virtual Tensor forward(const Tensor& tensor) override
        {
            Tensor img = tensor.reshape({ 256, 256, 3 });

            // encoder: keep each pre-pool activation as a skip connection
            Tensor s1, s2, s3, e;
            {
                GpuScope scope(_device, "enc1");
                s1 = enc1.forward(img).gelu();
            }
            {
                GpuScope scope(_device, "enc2");
                s2 = enc2.forward(pool.forward(s1)).gelu();
            }
            {
                GpuScope scope(_device, "enc3");
                s3 = enc3.forward(pool.forward(s2)).gelu();
            }
            {
                GpuScope scope(_device, "bottleneck");
                e = pool.forward(s3);
            }

            // decoder: upsample, concat the matching-resolution skip along channels, then conv
            Tensor d;
            {
                GpuScope scope(_device, "dec1");
                d = dec1.forward(upsample2x(e).cat(s3, 2)).gelu();
            }
            {
                GpuScope scope(_device, "dec2");
                d = dec2.forward(upsample2x(d).cat(s2, 2)).gelu();
            }
            {
                GpuScope scope(_device, "dec3");
                d = dec3.forward(upsample2x(d).cat(s1, 2));
            }

            return d.reshape({ 1, N });
        }

        virtual std::vector<Tensor> parameters() override
        {
            std::vector<Tensor> params;
            for (Module* layer : { static_cast<Module*>(&enc1), static_cast<Module*>(&enc2), static_cast<Module*>(&enc3),
                                   static_cast<Module*>(&dec1), static_cast<Module*>(&dec2), static_cast<Module*>(&dec3) })
            {
                std::vector<Tensor> p = layer->parameters();
                params.insert(params.end(), p.begin(), p.end());
            }
            return params;
        }

    private:
        // nearest-neighbour 2x upsample: (H,W,C) -> (2H,2W,C) via reshape + broadcast.
        // dual of the pooling downsample; differentiable through broadcast's backward.
        Tensor upsample2x(const Tensor& in)
        {
            const unsigned int H = in.shape()[0];
            const unsigned int W = in.shape()[1];
            const unsigned int C = in.shape()[2];
            return in.reshape({ H, 1, W, 1, C })
                .broadcast(1, 2)
                .broadcast(3, 2)
                .reshape({ H * 2, W * 2, C });
        }

        Device* _device;
        Conv2d enc1;
        Conv2d enc2;
        Conv2d enc3;
        Conv2d dec1;
        Conv2d dec2;
        Conv2d dec3;
        MaxPool2d pool;

    } autoencoder(device);

    std::filesystem::path model_path = "./model.bin";
    if (std::filesystem::exists(model_path))
    {
        // autoencoder.load(model_path);
    }

    Adam optimizer(autoencoder.parameters(), 0.001);

    std::filesystem::path opt_path = "./opt.bin";
    if (std::filesystem::exists(opt_path))
    {
        // optimizer.load(opt_path);
    }

    size_t steps = 1000;

    auto y = device->tensor(gt);

    for (size_t i = 1; i < steps; i++)
    {
        auto a = (device->rand(y.shape()) * 2.f - 1.f) + y;
        auto b = (device->rand(y.shape()) * 2.f - 1.f) + y;

        optimizer.zero_grad();

        Tensor L;
        {
            GpuScope step(device, "step");

            Tensor z;
            {
                GpuScope scope(device, "forward");
                z = autoencoder.forward(a);
            }

            L = z.mse(b);

            {
                GpuScope scope(device, "backward");
                L.backward();
            }
            {
                GpuScope scope(device, "optimizer");
                optimizer.step();
            }
        }

        device->submit();
        L.cpu([i, steps](std::vector<float> data)
              {
                  std::cout << "MSE " << data.front() << "\t" << i << "/" << steps << std::endl;
              });
    }
    std::cout << std::endl;

    auto a = (device->rand(y.shape()) * 2.f - 1.f) + y;
    auto z = autoencoder.forward(a);
    stbi_write_hdr("input.exr", 256, 256, 3, a.pow(2.2).cpu().data());
    stbi_write_hdr("output.exr", 256, 256, 3, z.pow(2.2).cpu().data());
    // autoencoder.save(model_path);
    // optimizer.save(opt_path);
    // }
    // catch (const std::exception& e)
    // {
    //     std::cerr << e.what() << '\n';
    // }
    return;
}

// ---------------------------------------------------------------------------
// Tensor op tests
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

static void check(const std::string& name, std::vector<float> got, std::vector<float> expected, float tol = 1e-3f)
{
    bool ok = got.size() == expected.size();
    for (size_t i = 0; ok && i < got.size(); i++)
    {
        if (std::isnan(got[i]) || std::fabs(got[i] - expected[i]) > tol)
        {
            ok = false;
        }
    }

    if (ok)
    {
        ++g_pass;
        std::cout << "[PASS] " << name << "\n";
        return;
    }

    ++g_fail;
    std::cout << "[FAIL] " << name << "\n      expected:";
    for (auto v : expected)
    {
        std::cout << " " << v;
    }
    std::cout << "\n      got:     ";
    for (auto v : got)
    {
        std::cout << " " << v;
    }
    std::cout << "\n";
}

int tensorTests()
{
    Instance instance;
    auto device = instance.device(1); // same GPU index learningSample() uses; change if needed

    const Shape s22 = { 2, 2 };

    // Fixed sequence operands (no rand).
    auto a = device->tensor({ 1, 2, 3, 4 }, s22);     // [[1,2],[3,4]]
    auto b = device->tensor({ 5, 6, 7, 8 }, s22);     // [[5,6],[7,8]]
    auto neg = device->tensor({ -1, 2, -3, 4 }, s22); // mixed signs (relu)
    auto v1 = device->tensor({ 1, 2, 3, 4 }, { 4 });  // 1-D
    auto v2 = device->tensor({ 5, 6, 7, 8 }, { 4 });  // 1-D

    // --- element-wise tensor/tensor ---
    check("add", (a + b).cpu(), { 6, 8, 10, 12 });
    check("sub", (a - b).cpu(), { -4, -4, -4, -4 });
    check("mul", (a * b).cpu(), { 5, 12, 21, 32 });
    check("div", (a / b).cpu(), { 0.2f, 0.333333f, 0.428571f, 0.5f });
    check("neg", (-a).cpu(), { -1, -2, -3, -4 });

    // --- scalar ops (member + free-function forms) ---
    check("add_scalar", (a + 10.f).cpu(), { 11, 12, 13, 14 });
    check("sub_scalar", (a - 1.f).cpu(), { 0, 1, 2, 3 });
    check("mul_scalar", (a * 2.f).cpu(), { 2, 4, 6, 8 });
    check("div_scalar", (a / 2.f).cpu(), { 0.5f, 1, 1.5f, 2 });
    check("radd", (10.f + a).cpu(), { 11, 12, 13, 14 });
    check("rsub", (10.f - a).cpu(), { 9, 8, 7, 6 });
    check("rmul", (2.f * a).cpu(), { 2, 4, 6, 8 });
    check("rdiv", (12.f / a).cpu(), { 12, 6, 4, 3 });
    check("lerp", lerp(a, b, 0.5f).cpu(), { 3, 4, 5, 6 });

    // --- shape / view ops ---
    check("reshape_1d", a.reshape({ 4 }).cpu(), { 1, 2, 3, 4 });
    check("reshape_row", a.reshape({ 1, 4 }).cpu(), { 1, 2, 3, 4 });
    check("permute_T", a.permute({ 1, 0 }).cpu(), { 1, 3, 2, 4 });
    check("mT", a.mT().cpu(), { 1, 3, 2, 4 });
    check("slice_0", a[0].cpu(), { 1, 2 });
    check("slice_1", a[1].cpu(), { 3, 4 });

    // --- linear algebra ---
    check("dot", v1.dot(v2).cpu(), { 70 });
    check("matmul", a.matmul(b).cpu(), { 19, 22, 43, 50 });
    {
        auto p = device->tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
        auto q = device->tensor({ 1, 2, 3, 4, 5, 6 }, { 3, 2 });
        check("matmul_nonsquare", p.matmul(q).cpu(), { 22, 28, 49, 64 });
    }
    {
        // tmatmul = this^T @ other (splits == 1 -> transposed-A read path).
        auto p = device->tensor({ 1, 2, 3, 4, 5, 6 }, { 3, 2 }); // (M=3, K=2)
        auto q = device->tensor({ 1, 0, 0, 1, 1, 1 }, { 3, 2 }); // (M=3, N=2)
        check("tmatmul", p.tmatmul(q).cpu(), { 6, 8, 8, 10 });
        check("tmatmul_eq_mT", p.tmatmul(q).cpu(), p.mT().matmul(q).cpu());
    }
    {
        // Split-K path: huge contraction, tiny output. With all-ones operands the
        // result of a K-contraction equals the contraction length (M).
        auto p = device->ones({ 16384, 2 }); // (M, K)
        auto q = device->ones({ 16384, 3 }); // (M, N)
        check("tmatmul_splitk", p.tmatmul(q).cpu(), { 16384, 16384, 16384, 16384, 16384, 16384 });

        auto r = device->ones({ 2, 16384 }); // (a, b) small output, huge contraction
        auto s = device->ones({ 16384, 3 }); // (b, c)
        check("matmul_splitk", r.matmul(s).cpu(), { 16384, 16384, 16384, 16384, 16384, 16384 });
    }

    // --- reductions / broadcast ---
    check("sum_global", a.sum().cpu(), { 10 });
    check("sum_dim0", a.sum(0).cpu(), { 4, 6 });
    check("sum_dim1", a.sum(1).cpu(), { 3, 7 });
    check("sum_dim1_keepdim", a.sum(1, true).cpu(), { 3, 7 });
    check("broadcast", a.sum(1, true).broadcast(1, 2).cpu(), { 3, 3, 7, 7 });

    // --- powers / roots ---
    check("pow_scalar", a.pow(2.f).cpu(), { 1, 4, 9, 16 });
    check("pow_tensor", a.pow(device->tensor({ 2 })).cpu(), { 1, 4, 9, 16 });
    check("sqrt", a.sqrt().cpu(), { 1, 1.414214f, 1.732051f, 2 });
    check("rcp", a.rcp().cpu(), { 1, 0.5f, 0.333333f, 0.25f });

    // --- transcendental (a = [1,2,3,4]) ---
    check("exp", a.exp().cpu(), { 2.718282f, 7.389056f, 20.085537f, 54.598150f });
    check("expm1", a.expm1().cpu(), { 1.718282f, 6.389056f, 19.085537f, 53.598150f });
    check("log", a.log().cpu(), { 0, 0.693147f, 1.098612f, 1.386294f });
    check("log1p", a.log1p().cpu(), { 0.693147f, 1.098612f, 1.386294f, 1.609438f });
    check("sin", a.sin().cpu(), { 0.841471f, 0.909297f, 0.141120f, -0.756802f });
    check("cos", a.cos().cpu(), { 0.540302f, -0.416147f, -0.989992f, -0.653644f });
    check("tan", a.tan().cpu(), { 1.557408f, -2.185040f, -0.142547f, 1.157821f });
    check("cosh", a.cosh().cpu(), { 1.543081f, 3.762196f, 10.067662f, 27.308233f });
    check("tanh", a.tanh().cpu(), { 0.761594f, 0.964028f, 0.995055f, 0.999329f });
    check("sech", a.sech().cpu(), { 0.648054f, 0.265802f, 0.099328f, 0.036619f });

    // --- activations ---
    check("relu", neg.relu().cpu(), { 0, 2, 0, 4 });
    check("leaky_relu", neg.relu(0.1f).cpu(), { -0.1f, 2, -0.3f, 4 });
    check("gelu", a.gelu().cpu(), { 0.841191f, 1.954792f, 2.996524f, 3.999857f });
    check("softmax", a.softmax().cpu(), { 0.032059f, 0.087144f, 0.236883f, 0.643914f });

    // --- loss ---
    check("mse", a.mse(b).cpu(), { 16 });

    // --- creation / misc ---
    check("zeros", device->zeros(s22).cpu(), { 0, 0, 0, 0 });
    check("ones", device->ones(s22).cpu(), { 1, 1, 1, 1 });
    check("repeat", device->repeat(5.f, s22).cpu(), { 5, 5, 5, 5 });
    check("detach", a.detach().cpu(), { 1, 2, 3, 4 });
    {
        auto dst = device->zeros(s22);
        dst.copy(a);
        check("copy", dst.cpu(), { 1, 2, 3, 4 });
    }

    // --- unfold (2,2,1) im2col with k=3, pad=1 -> (2,2,1,3,3) ---
    {
        auto u = device->tensor({ 1, 2, 3, 4 }, { 2, 2, 1 });
        check("unfold", u.unfold(3, 1).cpu(),
              { 0, 0, 0, 0, 1, 2, 0, 3, 4,
                0, 0, 0, 1, 2, 0, 3, 4, 0,
                0, 1, 2, 0, 3, 4, 0, 0, 0,
                1, 2, 0, 3, 4, 0, 0, 0, 0 });
    }

    // --- autograd: gradients accumulate into leaves ---
    {
        auto x = device->tensor({ 1, 2, 3, 4 }, s22);
        auto y = device->tensor({ 5, 6, 7, 8 }, s22);
        x.requires_grad(true); // leaves are no_grad by default; grads vanish without this
        y.requires_grad(true);
        auto z = x * y;
        z.backward();
        check("grad_mul_dx", x.grad().cpu(), { 5, 6, 7, 8 }); // dz/dx = y
        check("grad_mul_dy", y.grad().cpu(), { 1, 2, 3, 4 }); // dz/dy = x
    }
    {
        auto x = device->tensor({ 1, 2, 3, 4 }, s22);
        x.requires_grad(true);
        auto s = x.sum();
        s.backward();
        check("grad_sum", x.grad().cpu(), { 1, 1, 1, 1 }); // dsum/dx = 1
    }

    device->submit();

    std::cout << "\n"
              << g_pass << " passed, " << g_fail << " failed.\n";
    return g_fail == 0 ? 0 : 1;
}

int main()
{
    learningSample();
    // tensorTests();
    return 0;
}