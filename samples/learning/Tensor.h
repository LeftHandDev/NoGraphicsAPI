#ifndef SAMPLES_TENSOR_H
#define SAMPLES_TENSOR_H

#include <vector>
#include <string>
#include <optional>
#include <ostream>
#include <memory>

using Shape = std::vector<unsigned int>;

class Tensor_impl;
class Device_impl;
class Device;

template <typename T>
class Allocation;

class Tensor
{
public:
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    ~Tensor();

    operator std::string() const;

    std::vector<float> cpu();

    Shape shape() const;
    Tensor reshape(Shape) const;

    void copy(const Tensor&) const;
    void backward();

    Tensor operator+(const Tensor&) const; // element-wise addition
    Tensor operator-(const Tensor&) const; // element-wise subtraction
    Tensor operator*(const Tensor&) const; // element-wise multiplication
    Tensor operator/(const Tensor&) const; // element-wise division

    Tensor operator[](unsigned int) const; // takes a slice of a matrix

    Tensor mT() const;                  // 2D matrix transpose, +3D batched matrix transpose
    Tensor dot(const Tensor&) const;    // 1D dot product
    Tensor matmul(const Tensor&) const; // 2D matrix multiplication, +3D batched matrix multiplcation

    Tensor pow(const Tensor&) const;
    Tensor exp() const;
    Tensor tanh() const;

private:
    const float e = 2.718281828459045;
    Shape _shape;
    friend class Device_impl;
    explicit Tensor(Device_impl*, std::vector<float>, Shape = {}, bool slice = false);
    explicit Tensor(Device_impl*, Allocation<float>, Shape = {}, bool slice = false);
    std::unique_ptr<Tensor_impl> _tensor;
};

inline std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
    return os << static_cast<std::string>(t);
}

class Device
{
public:
    virtual ~Device() = default;
    virtual Tensor tensor(std::vector<float>, Shape = {}) = 0;
    virtual Tensor rand(Shape) = 0;
    virtual Tensor zeros(Shape) = 0;
    virtual Tensor ones(Shape) = 0;
    virtual Tensor repeat(float, Shape = {}) = 0;
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
};

class Layer : public Module
{
public:
    Layer(Device* device, unsigned int in, unsigned int out)
        : _weights(device->rand({ in, out })), _biases(device->zeros({ 1, out }))
    {
    }

    virtual Tensor forward(const Tensor& in) override
    {
        if (in.shape().size() == 1)
        {
            return (in.reshape({ 1, in.shape().front() }).matmul(_weights) + _biases).tanh();
        }
        return (in.matmul(_weights) + _biases).tanh();
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
        Tensor flow = _layers.front().forward(in);
        for (size_t i = 1; i < _layers.size(); i++)
        {
            flow = _layers[i].forward(flow);
        }
        return flow;
    }

private:
    std::vector<Layer> _layers;
};

#endif