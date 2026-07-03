#include "stb_image.h"
#include "stb_image_write.h"

#include "Learning.h"
#include "Tensor.h"

#include <iostream>
#include <random>
#include <chrono>

void learningSample()
{
    Instance instance;
    auto device = instance.device();

    // try
    // {
    std::vector<float> gt;
    int w, h, c;
    auto ptr = stbi_load("assets/Default.png", &w, &h, &c, 3);
    if (!ptr)
    {
        throw std::runtime_error(std::string("failed to load image: ") + stbi_failure_reason());
    }
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            gt.push_back(ptr[idx + 0] / 255.0f);
            gt.push_back(ptr[idx + 1] / 255.0f);
            gt.push_back(ptr[idx + 2] / 255.0f);
        }
    }
    stbi_image_free(ptr);

    const unsigned int N = 256 * 256 * 3;

    class Autoencoder : public Module
    {
    public:
        Autoencoder(Device* device)
            : enc1(device, 3, 16, 3, false, Tensor::Pad::reflect),
              enc2(device, 16, 4, 3, false, Tensor::Pad::reflect), // bottleneck: 4 channels
              dec1(device, 4, 16, 3, false, Tensor::Pad::reflect),
              dec2(device, 16, 3, 3, false, Tensor::Pad::reflect)
        {
        }

        virtual Tensor forward(const Tensor& tensor) override
        {
            Tensor img = tensor.reshape({ 256, 256, 3 });
            Tensor e = enc2.forward(enc1.forward(img).gelu()).gelu();
            Tensor d = dec2.forward(dec1.forward(e).gelu());
            return d.reshape({ 1, N });
        }

        virtual std::vector<Tensor> parameters() override
        {
            std::vector<Tensor> params;
            for (Module* layer : { static_cast<Module*>(&enc1), static_cast<Module*>(&enc2),
                                   static_cast<Module*>(&dec1), static_cast<Module*>(&dec2) })
            {
                std::vector<Tensor> p = layer->parameters();
                params.insert(params.end(), p.begin(), p.end());
            }
            return params;
        }

    private:
        Conv2d enc1;
        Conv2d enc2;
        Conv2d dec1;
        Conv2d dec2;

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

    size_t steps = 10000;
    auto y = device->tensor({ gt });

    float noise_ratio = 0.25;

    auto shape = y.shape();
    unsigned int batch = 1; // Conv2d processes a single (H,W,C) image
    shape.insert(shape.begin(), { batch });

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < steps; i++)
    {
        auto a = (device->rand(shape) * 2.f - 1.f) * noise_ratio + y;
        auto b = (device->rand(shape) * 2.f - 1.f) * noise_ratio + y;

        optimizer.zero_grad();
        auto z = autoencoder.forward(a.detach());
        auto L = z.mse(b.detach());
        L.backward();
        optimizer.step();
        device->submit();
        L.cpu([&](std::vector<float> data)
              { if (i >= steps) return; std::cout << "MSE " << data.front() << "\t" << i << "/" << steps << "\t" << std::endl; });
    }
    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now() - start;

    std::cout << "Training took: " << std::chrono::duration_cast<std::chrono::seconds>(end).count() << " seconds";

    auto a = (device->rand(y.shape()) * 2.f - 1.f) * noise_ratio + y;
    auto z = autoencoder.forward(a);
    stbi_write_hdr("input.exr", 256, 256, 3, a.pow(2.2).cpu().data());
    stbi_write_hdr("output.exr", 256, 256, 3, z.pow(2.2).cpu().data());
    stbi_write_hdr("gt.exr", 256, 256, 3, y.pow(2.2).cpu().data());
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
        auto z = x * y;
        z.backward();
        check("grad_mul_dx", x.grad().cpu(), { 5, 6, 7, 8 }); // dz/dx = y
        check("grad_mul_dy", y.grad().cpu(), { 1, 2, 3, 4 }); // dz/dy = x
    }
    {
        auto x = device->tensor({ 1, 2, 3, 4 }, s22);
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
    return 0;
}