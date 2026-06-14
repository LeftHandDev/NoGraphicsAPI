#ifndef SAMPLES_TENSOR_H
#define SAMPLES_TENSOR_H

#include <vector>
#include <string>
#include <optional>
#include <ostream>
#include <memory>
#include <functional>
#include <set>

using Shape = std::vector<unsigned int>;

class Tensor_impl;
class Device_impl;
class Device;

template <typename T>
class Allocation;

class Tensor
{
public:
    Tensor() = default;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor();

    operator std::string() const;

    std::vector<float> cpu();

    bool null() const;
    Shape shape() const;
    Tensor grad() const;
    Tensor reshape(Shape) const;
    Tensor detach() const;

    Tensor repeat(const Tensor&, Shape) const;
    void copy(const Tensor&) const;
    void backward();

    Tensor operator-() const; // element-wise negative

    Tensor operator+(const Tensor&) const; // element-wise addition
    Tensor operator-(const Tensor&) const; // element-wise subtraction
    Tensor operator*(const Tensor&) const; // element-wise multiplication
    Tensor operator/(const Tensor&) const; // element-wise division

    Tensor operator+(float) const; // scalar addition
    Tensor operator-(float) const; // scalar subtraction
    Tensor operator*(float) const; // scalar multiplication
    Tensor operator/(float) const; // scalar division

    Tensor operator[](unsigned int) const; // takes a slice of a matrix

    Tensor mT() const;                  // 2D matrix transpose, +3D batched matrix transpose
    Tensor dot(const Tensor&) const;    // 1D dot product
    Tensor matmul(const Tensor&) const; // 2D matrix multiplication, +3D batched matrix multiplcation

    Tensor pow(const Tensor&) const;
    Tensor pow(float) const;

    Tensor sum() const;
    Tensor sqrt() const;
    Tensor rcp() const;
    Tensor exp() const;
    Tensor cosh() const;
    Tensor tanh() const;
    Tensor sech() const;
    Tensor relu(float = 0.f) const;
    Tensor gelu() const;

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
    friend class Device_impl;
    explicit Tensor(Device_impl*, std::vector<float>, std::vector<Tensor> prev, Shape = {}, bool slice = false);
    explicit Tensor(Device_impl*, Allocation<float>, std::vector<Tensor> prev, Shape = {}, bool slice = false);
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

inline std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    return os << static_cast<std::string>(t);
}

class Device
{
public:
    virtual ~Device() = default;
    virtual void submit() = 0;
    virtual Tensor tensor(std::vector<float>, Shape = {}) = 0;
    virtual Tensor rand(Shape) = 0;
    virtual Tensor zeros(Shape) = 0;
    virtual Tensor ones(Shape) = 0;
    virtual Tensor repeat(float, Shape) = 0;
    virtual Tensor repeat(const Tensor&, Shape) = 0;
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
    virtual Tensor forward(const Tensor& in) = 0;
    virtual void train(float lr) = 0;
};

class Layer : public Module
{
public:
    Layer(Device* device, unsigned int in, unsigned int out)
        : _weights(((device->rand({ in, out }) * 2.f - 1.f) * sqrt(1.f / in)).detach()), _biases(device->zeros({ 1, out }))
    {
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
        return in.matmul(_weights) + _biases;
    }

    virtual void train(float lr) override
    {
        _weights = (_weights - _weights.grad() * lr).detach();
        _biases = (_biases - _biases.grad() * lr).detach();
    }

private:
    Tensor _weights;
    Tensor _biases;
};

class Network : public Module
{
public:
    Network(Device* device, Shape layers)
    {
        for (size_t i = 1; i < layers.size(); i++)
        {
            _layers.push_back({ device, layers[i - 1], layers[i] });
        }
    }

    virtual Tensor forward(const Tensor& in) override
    {
        Tensor out = _layers.front().forward(in);
        for (size_t i = 1; i < _layers.size(); i++)
        {
            out = _layers[i].forward(out.gelu()); //.relu(0.01f));
        }
        return out;
    }

    virtual void train(float lr) override
    {
        for (auto& layer : _layers)
        {
            layer.train(lr);
        }
    }

private:
    std::vector<Layer> _layers;
};

#endif