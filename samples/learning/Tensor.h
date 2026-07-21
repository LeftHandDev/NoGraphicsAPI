#ifndef SAMPLES_TENSOR_H
#define SAMPLES_TENSOR_H

#include <vector>
#include <string>
#include <optional>
#include <ostream>
#include <memory>
#include <functional>
#include <set>
#include <filesystem>

using Shape = std::vector<unsigned int>;

class Tensor_impl;
class Device_impl;
class Device;

template <typename T>
class Allocation;

class Tensor
{
public:
    enum class Type
    {
        float16,
        float32
    };

    enum class Pad
    {
        zero,
        reflect
    };

    Tensor() = default;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor();

    operator std::string() const;

    Device* device();

    std::vector<float> cpu();                          // blocking
    void cpu(std::function<void(std::vector<float>)>); // non-blocking

    bool null() const;
    void zero() const; // zero the grad

    bool requires_grad() const;
    Tensor& requires_grad(bool value); // mark this tensor as a grad-tracking leaf

    Shape shape() const;
    Type type() const;
    uint64_t numel() const;

    Tensor grad() const;
    Tensor reshape(Shape) const;
    Tensor permute(Shape) const;
    Tensor detach() const; // create a clone detached from the graph

    Tensor repeat(const Tensor&, Shape) const;
    Tensor unfold(unsigned int, unsigned int, Pad = Pad::zero, unsigned int stride = 1) const; // im2col: (H,W,C) -> (OH,OW,C,k,k), zero or reflect padded, strided

    void copy(const Tensor&) const; // copy from
    void backward();

    Tensor operator-() const; // element-wise negation

    Tensor operator+(const Tensor&) const; // element-wise addition
    Tensor operator-(const Tensor&) const; // element-wise subtraction
    Tensor operator*(const Tensor&) const; // element-wise multiplication
    Tensor operator/(const Tensor&) const; // element-wise division

    Tensor operator+(float) const; // scalar addition
    Tensor operator-(float) const; // scalar subtraction
    Tensor operator*(float) const; // scalar multiplication
    Tensor operator/(float) const; // scalar division

    Tensor operator[](unsigned int) const; // take a slice of a tensor

    Tensor mT() const;                                 // 2D matrix transpose, +3D batched matrix transpose
    Tensor dot(const Tensor&) const;                   // 1D dot product
    Tensor matmul(const Tensor&) const;                // 2D matrix multiplication, +3D batched matrix multiplcation
    Tensor tmatmul(const Tensor&) const;               // fused this^T @ other (no materialized transpose); used by matmul backward
    Tensor affine(const Tensor&, const Tensor&) const; // Use wmma when possible, otherwise falls back to matmul(weights) + biases

    Tensor pow(const Tensor&) const;
    Tensor pow(float) const;

    Tensor mse(const Tensor&) const;
    Tensor sum() const;
    Tensor sum(int dim, bool keepdim = false) const;    // reduce-sum over a single axis
    Tensor max(int dim, bool keepdim = false) const;    // reduce-max over a single axis
    Tensor broadcast(int dim, unsigned int size) const; // repeat a size-1 axis (dual of sum)
    Tensor cat(const Tensor&, int dim) const;           // concatenate two tensors along an axis
    Tensor sqrt() const;
    Tensor rcp() const;
    Tensor exp() const;
    Tensor expm1() const;
    Tensor log() const;
    Tensor log1p() const;
    Tensor sin() const;
    Tensor cos() const;
    Tensor tan() const;
    Tensor cosh() const;
    Tensor tanh() const;
    Tensor sech() const;
    Tensor relu(float = 0.f) const;
    Tensor gelu() const;
    Tensor softmax() const;

    Tensor float16() const;
    Tensor float32() const;

    // usage: weights = (weights - lr * grad.adam(mean, variance, step));
    Tensor adam(Tensor& mean, Tensor& variance, uint64_t steps, float b1 = 0.9, float b2 = 0.999);

    bool operator==(const Tensor& other) const
    {
        return _self < other._self;
    }

    bool operator<(const Tensor& other) const
    {
        return _self < other._self;
    }

private:
    const float e = 2.718281828459045f;
    const float pi = 3.1415926535f;
    const Shape unit = { 1 };
    Shape _shape;
    Type _type;
    friend class Device_impl;
    explicit Tensor(Device_impl*, std::vector<float>, std::vector<Tensor> prev, Shape = {}, Type = Type::float32, bool slice = false);
    explicit Tensor(Device_impl*, Allocation<uint8_t>, std::vector<Tensor> prev, Shape = {}, Type = Type::float32, bool slice = false);
    std::shared_ptr<Tensor_impl> _self;
    static void build(Tensor, std::set<Tensor>&, std::vector<Tensor>&);
};

inline Tensor operator+(float x, Tensor t)
{
    return t + x;
}

inline Tensor operator-(float x, Tensor t)
{
    return -t + x;
}

inline Tensor operator*(float x, Tensor t)
{
    return t * x;
}

inline Tensor operator/(float x, Tensor t)
{
    return x * t.rcp();
}

inline Tensor lerp(const Tensor& a, const Tensor& b, float t)
{
    return (1.f - t) * a + t * b;
}

inline std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    return os << static_cast<std::string>(t);
}

// Scope disable grad tracking
class NoGrad
{
public:
    NoGrad();
    ~NoGrad();
    NoGrad(const NoGrad&) = delete;
    NoGrad& operator=(const NoGrad&) = delete;

private:
    bool _previous;
};

class Device
{
public:
    virtual ~Device() = default;
    virtual void submit() = 0;
    virtual Tensor tensor(std::vector<float>, Shape = {}, Tensor::Type = Tensor::Type::float32) = 0;
    virtual Tensor rand(Shape, Tensor::Type = Tensor::Type::float32) = 0;
    virtual Tensor zeros(Shape, Tensor::Type = Tensor::Type::float32) = 0;
    virtual Tensor ones(Shape, Tensor::Type = Tensor::Type::float32) = 0;
    virtual Tensor repeat(float, Shape) = 0;
    virtual Tensor repeat(const Tensor&, Shape) = 0;

    virtual void pushMarker(const char* name) = 0;
    virtual void popMarker() = 0;
};

class GpuScope
{
public:
    GpuScope(Device* device, const char* name) : _device(device)
    {
        _device->pushMarker(name);
    }
    ~GpuScope()
    {
        _device->popMarker();
    }
    GpuScope(const GpuScope&) = delete;
    GpuScope& operator=(const GpuScope&) = delete;

private:
    Device* _device;
};

class Instance
{
public:
    Instance();

    ~Instance();

    Device* device(int index = 0);
};

class Module
{
public:
    ~Module()
    {
    }
    virtual Tensor forward(const Tensor&) = 0;
    virtual std::vector<Tensor> parameters() = 0;
    void save(std::filesystem::path);
    void load(std::filesystem::path);
};

class Linear : public Module
{
public:
    Linear(Device* device, unsigned int in, unsigned int out, bool affine)
        : _weights((device->rand({ in, out }) * 2.f - 1.f) * sqrt(1.f / in)),
          _biases(device->zeros({ 1, out }))
    {
        if (affine)
        {
            _weights = _weights.float16();
        }

        _weights.requires_grad(true);
        _biases.requires_grad(true);
    }

    virtual Tensor forward(const Tensor& in) override
    {
        if (in.shape().size() > 2)
        {
            throw std::runtime_error("Unable to forward tensor with more than 2 dimensions");
        }

        if (in.shape().size() == 1)
        {
            return in.reshape({ 1, in.shape().front() }).matmul(_weights) + _biases;
        }
        return _weights.type() == Tensor::Type::float16 ? in.affine(_weights, _biases) : in.matmul(_weights) + _biases;
    }

    virtual std::vector<Tensor> parameters() override
    {
        return { _weights, _biases };
    }

private:
    Tensor _weights;
    Tensor _biases;
};

class Sequential : public Module
{
public:
    Sequential(Device* device, Shape layers, bool affine = false)
    {
        for (size_t i = 1; i < layers.size(); i++)
        {
            _layers.push_back({ device, layers[i - 1], layers[i], affine });
        }
    }

    virtual Tensor forward(const Tensor& in) override
    {
        Tensor out = _layers.front().forward(in);
        for (size_t i = 1; i < _layers.size(); i++)
        {
            out = _layers[i].forward(out.gelu());
        }
        return out;
    }

    virtual std::vector<Tensor> parameters() override
    {
        std::vector<Tensor> params;
        for (auto layer : _layers)
        {
            for (auto& tensor : layer.parameters())
            {
                params.push_back(tensor);
            }
        }
        return params;
    }

private:
    std::vector<Linear> _layers;
};

class Conv2d : public Module
{
public:
    // 2D convolution with "same"-style k/2 padding. Input (H,W,C) -> output (OH,OW,out).
    Conv2d(Device* device, int in_channels, int out_channels, int kernel_size, bool affine = false, Tensor::Pad pad = Tensor::Pad::zero, int stride = 1)
        : _in_channels(in_channels),
          _out_channels(out_channels),
          _kernel_size(kernel_size),
          _stride(stride),
          _pad(pad),
          _weights((device->rand({ static_cast<unsigned int>(in_channels * kernel_size * kernel_size), static_cast<unsigned int>(out_channels) }) * 2.f - 1.f) * sqrt(1.f / (in_channels * kernel_size * kernel_size))),
          _biases(device->zeros({ 1, static_cast<unsigned int>(out_channels) }))
    {
        if (affine)
        {
            _weights = _weights.float16();
        }

        _weights.requires_grad(true);
        _biases.requires_grad(true);
    }

    virtual Tensor forward(const Tensor& in) override
    {
        if (in.shape().size() != 3)
        {
            throw std::runtime_error("Conv2d expects a 3D (H, W, C) tensor");
        }

        const unsigned int H = in.shape()[0];
        const unsigned int W = in.shape()[1];
        const unsigned int C = in.shape()[2];

        if (C != static_cast<unsigned int>(_in_channels))
        {
            throw std::runtime_error("Conv2d input channels mismatch");
        }

        const unsigned int patch = static_cast<unsigned int>(_in_channels * _kernel_size * _kernel_size);
        const unsigned int out = static_cast<unsigned int>(_out_channels);
        const unsigned int pad = static_cast<unsigned int>(_kernel_size / 2);
        const unsigned int stride = static_cast<unsigned int>(_stride);

        const unsigned int OH = (H + 2 * pad - _kernel_size) / stride + 1;
        const unsigned int OW = (W + 2 * pad - _kernel_size) / stride + 1;

        Tensor cols = in.unfold(_kernel_size, pad, _pad, stride).reshape({ OH * OW, patch });

        Tensor result = _weights.type() == Tensor::Type::float16 ? cols.affine(_weights, _biases) : cols.matmul(_weights) + _biases;

        return result.reshape({ OH, OW, out });
    }

    virtual std::vector<Tensor> parameters() override
    {
        return { _weights, _biases };
    }

private:
    int _in_channels;
    int _out_channels;
    int _kernel_size;
    int _stride;
    Tensor::Pad _pad;
    Tensor _weights;
    Tensor _biases;
};

class MaxPool2d : public Module
{
public:
    MaxPool2d(int kernel_size, int stride = 0)
        : _kernel_size(kernel_size),
          _stride(stride > 0 ? stride : kernel_size)
    {
    }

    virtual Tensor forward(const Tensor& in) override
    {
        if (in.shape().size() != 3)
        {
            throw std::runtime_error("MaxPool2d expects a 3D (H, W, C) tensor");
        }

        const unsigned int H = in.shape()[0];
        const unsigned int W = in.shape()[1];
        const unsigned int C = in.shape()[2];
        const unsigned int k = static_cast<unsigned int>(_kernel_size);
        const unsigned int stride = static_cast<unsigned int>(_stride);

        const unsigned int OH = (H - k) / stride + 1;
        const unsigned int OW = (W - k) / stride + 1;

        Tensor cols = in.unfold(_kernel_size, 0, Tensor::Pad::zero, stride).reshape({ OH * OW * C, k * k });
        return cols.max(1).reshape({ OH, OW, C });
    }

    virtual std::vector<Tensor> parameters() override
    {
        return {};
    }

private:
    int _kernel_size;
    int _stride;
};

class Optimizer
{
public:
    Optimizer(std::vector<Tensor> parameters) : _parameters(parameters)
    {
    }

    ~Optimizer()
    {
    }

    virtual void zero_grad()
    {
        for (auto p : _parameters)
        {
            p.zero();
        }
    }

    virtual void step() = 0;

protected:
    std::vector<Tensor> _parameters;
};

class SGD : public Optimizer
{
public:
    SGD(std::vector<Tensor> parameters, float lr) : Optimizer(parameters), _lr(lr)
    {
    }

    virtual void step() override
    {
        NoGrad no_grad;
        for (auto& p : _parameters)
        {
            p.copy(p - _lr * p.grad());
        }
    }

private:
    float _lr;
};

class Adam : public Optimizer
{
public:
    Adam(std::vector<Tensor> parameters, float lr) : Optimizer(parameters), _lr(lr)
    {
        NoGrad no_grad;
        for (auto p : parameters)
        {
            _mean.push_back(p * 0.f);
            _variance.push_back(p * 0.f);
        }
    }

    virtual void step() override
    {
        NoGrad no_grad;
        ++_steps;
        for (size_t i = 0; i < _parameters.size(); i++)
        {
            _parameters[i].copy(_parameters[i] - _lr * _parameters[i].grad().adam(_mean[i], _variance[i], _steps));
        }
    }

    void save(std::filesystem::path path);
    void load(std::filesystem::path path);

private:
    float _lr;
    std::vector<Tensor> _mean;
    std::vector<Tensor> _variance;
    uint64_t _steps = 0;
};

#endif