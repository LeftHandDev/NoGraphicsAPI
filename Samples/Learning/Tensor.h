#ifndef SAMPLES_TENSOR_H
#define SAMPLES_TENSOR_H

#include <vector>
#include <string>
#include <optional>
#include <ostream>
#include <memory>

using shape = std::vector<unsigned int>;

class tensor_impl;
class device_impl;
class device;

class tensor
{
public:
    tensor(tensor &&) noexcept;
    tensor &operator=(tensor &&) noexcept;
    ~tensor();

    operator std::string() const;

    std::vector<float> cpu();

    shape shape() const;
    tensor reshape(::shape) const;

    void backward();

    tensor operator+(const tensor &) const; // element-wise addition
    tensor operator-(const tensor &) const; // element-wise subtraction
    tensor operator*(const tensor &) const; // element-wise multiplication
    tensor operator/(const tensor &) const; // element-wise division

    tensor operator[](unsigned int) const; // takes a slice of a matrix

    tensor mT() const;                   // 2D matrix transpose, +3D batched matrix transpose
    tensor dot(const tensor &) const;    // 1D dot product
    tensor matmul(const tensor &) const; // 2D matrix multiplication, +3D batched matrix multiplcation

    tensor pow(const tensor &) const;
    tensor exp() const;
    tensor tanh() const;

private:
    const float e = 2.718281828459045;
    ::shape _shape;
    friend class device_impl;
    explicit tensor(device_impl *, std::vector<float>, ::shape = {});
    explicit tensor(tensor_impl, ::shape = {});
    std::unique_ptr<tensor_impl> _tensor;
};

inline std::ostream &operator<<(std::ostream &os, const tensor &t)
{
    return os << static_cast<std::string>(t);
}

class device
{
public:
    virtual tensor tensor(std::vector<float>, shape = {}) = 0;
    virtual ::tensor rand(::shape) = 0;
    virtual ::tensor zeros(::shape) = 0;
    virtual ::tensor ones(::shape) = 0;
    virtual ::tensor repeat(float, ::shape = {}) = 0;
};

class instance
{
public:
    instance();

    ~instance();

    device *device();
};

class module
{
public:
    ~module() {}
    virtual tensor forward(tensor) = 0;
};

class layer : public module
{
public:
    layer(device *device, unsigned int in, unsigned int out)
        : _weights(device->rand({in, out})), _biases(device->zeros({1, out}))
    {
    }

    virtual tensor forward(tensor in) override
    {
        if (in.shape().size() == 1)
        {
            in = in.reshape({1, in.shape().front()});
        }
        return (in.matmul(_weights) + _biases).tanh();
    }

private:
    tensor _weights;
    tensor _biases;
};

class network : public module
{
public:
    network(device *device, shape layers)
    {
        for (size_t i = 1; i < layers.size(); i++)
        {
            _layers.push_back({device, layers[i - 1], layers[i]});
        }
    }

    virtual tensor forward(tensor in) override
    {
        tensor flow = _layers.front().forward(std::move(in));
        for (size_t i = 1; i < _layers.size(); i++)
        {
            flow = _layers[i].forward(std::move(flow));
        }
        return flow;
    }

private:
    std::vector<layer> _layers;
};

#endif