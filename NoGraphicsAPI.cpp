#include "NoGraphicsAPI.h"
#include "VkBootstrap.h"

#include <map>
#include <vector>

struct GpuPipeline_T 
{ 
    VkPipeline pipeline; 
    VkPipelineLayout layout; 
    VkDescriptorSetLayout setLayout; 
    VkPipelineBindPoint bindPoint;
};
struct GpuTexture_T { GpuTextureDesc desc; };
struct GpuDepthStencilState_T {  };
struct GpuBlendState_T { };
struct GpuQueue_T { VkQueue queue; };
struct GpuCommandBuffer_T { VkCommandBuffer commandBuffer; };
struct GpuSemaphore_T { VkSemaphore semaphore; };

struct Allocation
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* ptr = nullptr;
};

struct Vulkan
{
    // VkBootstrap objects
    vkb::Instance instance;
    vkb::InstanceDispatchTable instanceDispatchTable;
    vkb::PhysicalDevice physicalDevice;
    vkb::Device device;
    vkb::DispatchTable dispatchTable;

    // Vulkan objects
    VkCommandPool commandPool;
    std::map<std::pair<VkSemaphore, uint64_t>, std::vector<VkCommandBuffer>> submittedCommandBuffers;

    // Vulkan structs
    VkPhysicalDeviceMemoryProperties memoryProperties = {};

    // Allocation tracking
    std::vector<Allocation> allocations;

    // Pipelines
    GpuPipeline currentPipeline = nullptr;

    Vulkan()
    {
        vkb::InstanceBuilder instanceBuilder;
        auto instanceRet = instanceBuilder.request_validation_layers(true)
            .require_api_version(VK_API_VERSION_1_4)
            .build();
        instance = instanceRet.value();
        instanceDispatchTable = instance.make_table();

        vkb::PhysicalDeviceSelector deviceSelector{ instance };
        auto physicalDeviceRet = deviceSelector.add_required_extensions({
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
        }).defer_surface_initialization().select();
        physicalDevice = physicalDeviceRet.value();

        VkPhysicalDeviceVulkan14Features physicalDeviceVulkan14Features = {};
        physicalDeviceVulkan14Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        physicalDeviceVulkan14Features.pushDescriptor = VK_TRUE;

        VkPhysicalDeviceVulkan12Features physicalDeviceVulkan12Features = {};
        physicalDeviceVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        physicalDeviceVulkan12Features.pNext = &physicalDeviceVulkan14Features;
        physicalDeviceVulkan12Features.timelineSemaphore = VK_TRUE;
        physicalDeviceVulkan12Features.bufferDeviceAddress = VK_TRUE;

        physicalDevice.enable_extension_features_if_present(physicalDeviceVulkan12Features);
        instanceDispatchTable.getPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        vkb::DeviceBuilder deviceBuilder{ physicalDevice };
        auto deviceRet = deviceBuilder.build();
        device = deviceRet.value();
        dispatchTable = device.make_table();

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = device.get_queue_index(vkb::QueueType::graphics).value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        dispatchTable.createCommandPool(&poolInfo, nullptr, &commandPool);
    }

    ~Vulkan()
    {
        dispatchTable.deviceWaitIdle();
        dispatchTable.destroyCommandPool(commandPool, nullptr);
        vkb::destroy_device(device);
        vkb::destroy_instance(instance);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        return UINT32_MAX;
    }

    Allocation findAllocation(VkDeviceAddress address)
    {
        for (auto& buffer : allocations)
        {
            if (address >= buffer.address && address < (buffer.address + buffer.size))
            {    
                return buffer;
            }
        }

        return {};
    }

    Allocation findAllocation(void* ptr)
    {
        for (auto& buffer : allocations)
        {
            if (ptr >= buffer.ptr && ptr < (static_cast<uint8_t*>(buffer.ptr) + buffer.size))
            {    
                return buffer;
            }
        }

        return {};
    }

    void freeAllocation(Allocation& alloc)
    {
        if (alloc.ptr)
        {
            dispatchTable.unmapMemory(alloc.memory);
        }

        dispatchTable.freeMemory(alloc.memory, nullptr);
        dispatchTable.destroyBuffer(alloc.buffer, nullptr);

        allocations.erase(std::remove_if(allocations.begin(), allocations.end(),
            [alloc](const Allocation& b) { return b.buffer == alloc.buffer; }), allocations.end());
    }

    Allocation createAllocation(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceSize alignment)
    {
        Allocation alloc = { .size = size };

        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.queueFamilyIndexCount = 0;
        dispatchTable.createBuffer(&bufferInfo, nullptr, &alloc.buffer);

        VkMemoryRequirements memRequirements = {};
        dispatchTable.getBufferMemoryRequirements(alloc.buffer, &memRequirements);

        alignment = std::max(alignment, memRequirements.alignment);
        VkDeviceSize alignedSize = (memRequirements.size + alignment - 1) & ~(alignment - 1);

        VkMemoryAllocateFlagsInfo allocateFlagsInfo = {};
        allocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        allocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext = &allocateFlagsInfo;
        allocInfo.allocationSize = alignedSize;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        dispatchTable.allocateMemory(&allocInfo, nullptr, &alloc.memory);
        dispatchTable.bindBufferMemory(alloc.buffer, alloc.memory, 0);

        VkBufferDeviceAddressInfo addressInfo = {};
        addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addressInfo.buffer = alloc.buffer;
        alloc.address = dispatchTable.getBufferDeviceAddress(&addressInfo);

        VkDeviceSize offset = (alignment - (alloc.address % alignment)) % alignment;
        alloc.address += offset;

        if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        {
            dispatchTable.mapMemory(alloc.memory, 0, alignedSize, 0, &alloc.ptr);
            alloc.ptr = static_cast<uint8_t*>(alloc.ptr) + offset;
        }

        allocations.push_back(alloc);

        return alloc;
    }

    VkImage createImage(GpuTextureDesc desc)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = desc.type == TEXTURE_1D ? VK_IMAGE_TYPE_1D :
                            desc.type == TEXTURE_2D ? VK_IMAGE_TYPE_2D :
                            desc.type == TEXTURE_3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_MAX_ENUM;
        imageInfo.extent.width = desc.dimensions.x;
        imageInfo.extent.height = desc.dimensions.y;
        imageInfo.extent.depth = desc.dimensions.z;
        imageInfo.mipLevels = desc.mipCount;
        imageInfo.arrayLayers = desc.layerCount;
        imageInfo.samples = static_cast<VkSampleCountFlagBits>(desc.sampleCount);
        imageInfo.format = desc.format == FORMAT_RGBA8_UNORM ? VK_FORMAT_R8G8B8A8_UNORM :
                        desc.format == FORMAT_D32_FLOAT ? VK_FORMAT_D32_SFLOAT :
                        desc.format == FORMAT_RG11B10_FLOAT ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 :
                        desc.format == FORMAT_RGB10_A2_UNORM ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_UNDEFINED;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = desc.usage == USAGE_SAMPLED ? VK_IMAGE_USAGE_SAMPLED_BIT :
                        desc.usage == USAGE_STORAGE ? VK_IMAGE_USAGE_STORAGE_BIT :
                        desc.usage == USAGE_COLOR_ATTACHMENT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT :
                        desc.usage == USAGE_DEPTH_STENCIL_ATTACHMENT ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkMemoryRequirements imageMemoryRequirements = {};

        VkImage image;
        dispatchTable.createImage(&imageInfo, nullptr, &image);

        return image;
    }
};

static Vulkan vulkan;

void* gpuMalloc(size_t bytes, MEMORY memory)
{
    return gpuMalloc(bytes, GPU_DEFAULT_ALIGNMENT, memory);
}

void* gpuMalloc(size_t bytes, size_t align, MEMORY memory)
{
    switch (memory)
    {
    case MEMORY_DEFAULT: // DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
    {
        auto usage = 
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | 
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

        auto properties = 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        auto alloc = vulkan.createAllocation(bytes, usage, properties, align);
        return alloc.ptr;        
    }
    case MEMORY_GPU: // DEVICE_LOCAL
    {
        auto usage = 
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | 
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

        auto properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        auto alloc = vulkan.createAllocation(bytes, usage, properties, align);
        return reinterpret_cast<void*>(alloc.address);
    }
    case MEMORY_READBACK: // HOST_VISIBLE | HOST_COHERENT | HOST_CACHED
    {
        auto usage = 
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | 
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        auto properties = 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        auto alloc = vulkan.createAllocation(bytes, usage, properties, align);
        return alloc.ptr;    
    }
    };

    return nullptr;
}

void gpuFree(void *ptr)
{
    for (auto& alloc : vulkan.allocations)
    {
        if (alloc.ptr == ptr || reinterpret_cast<VkDeviceAddress>(ptr) == alloc.address)
        {
            vulkan.freeAllocation(alloc);
            return;
        }
    }
}

void* gpuHostToDevicePointer(void *ptr)
{
    Allocation alloc = vulkan.findAllocation(ptr);
    if (alloc.buffer != VK_NULL_HANDLE)
    {
        return reinterpret_cast<void*>(alloc.address + (static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(alloc.ptr)));
    }
    return nullptr;
}

GpuTextureSizeAlign gpuTextureSizeAlign(GpuTextureDesc desc)
{
    VkImage image = vulkan.createImage(desc);

    VkMemoryRequirements imageMemoryRequirements = {};
    vulkan.dispatchTable.getImageMemoryRequirements(image, &imageMemoryRequirements);
    vulkan.dispatchTable.destroyImage(image, nullptr);

    return { imageMemoryRequirements.size, imageMemoryRequirements.alignment };
}

GpuTexture gpuCreateTexture(GpuTextureDesc desc, void* ptrGpu)
{
    Allocation alloc = vulkan.findAllocation(reinterpret_cast<VkDeviceAddress>(ptrGpu));

    if (alloc.buffer == VK_NULL_HANDLE)
    {
        return nullptr;
    }

    alloc.image = vulkan.createImage(desc);

    VkDeviceSize offset = reinterpret_cast<VkDeviceAddress>(ptrGpu) - alloc.address;
    vulkan.dispatchTable.bindImageMemory(alloc.image, alloc.memory, offset);

    return new GpuTexture_T { desc };
}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc)
{
    return {};
}

GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc)
{
    return {};
}

GpuPipeline gpuCreateComputePipeline(ByteSpan computeIR)
{
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = computeIR.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(computeIR.data());
    VkShaderModule shaderModule;
    vulkan.dispatchTable.createShaderModule(&shaderModuleCreateInfo, nullptr, &shaderModule);

    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
    descriptorSetLayoutCreateInfo.bindingCount = 1;
    descriptorSetLayoutCreateInfo.pBindings = &binding;
    VkDescriptorSetLayout descriptorSetLayout;
    vulkan.dispatchTable.createDescriptorSetLayout(&descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    vulkan.dispatchTable.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pName = "main";

    VkPipeline pipeline;
    vulkan.dispatchTable.createComputePipelines(VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);
    vulkan.dispatchTable.destroyShaderModule(shaderModule, nullptr);

    return new GpuPipeline_T { pipeline, pipelineLayout, descriptorSetLayout, VK_PIPELINE_BIND_POINT_COMPUTE };
}

GpuPipeline gpuCreateGraphicsPipeline(ByteSpan vertexIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    return nullptr;
}

GpuPipeline gpuCreateGraphicsMeshletPipeline(ByteSpan meshletIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    return nullptr;
}

void gpuFreePipeline(GpuPipeline pipeline)
{
    vulkan.dispatchTable.destroyPipeline(pipeline->pipeline, nullptr);
    vulkan.dispatchTable.destroyPipelineLayout(pipeline->layout, nullptr);
    vulkan.dispatchTable.destroyDescriptorSetLayout(pipeline->setLayout, nullptr);
    delete pipeline;
}

GpuDepthStencilState gpuCreateDepthStencilState(GpuDepthStencilDesc desc)
{
    return nullptr;
}

GpuBlendState gpuCreateBlendState(GpuBlendDesc desc)
{
    return nullptr;
}

void gpuFreeDepthStencilState(GpuDepthStencilState state)
{
}

void gpuFreeBlendState(GpuBlendState state)
{
}

GpuQueue gpuCreateQueue()
{
    return new GpuQueue_T { vulkan.device.get_queue(vkb::QueueType::graphics).value() };
}

void gpuFreeQueue(GpuQueue queue)
{
    delete queue;
}

GpuCommandBuffer gpuStartCommandRecording(GpuQueue queue)
{
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vulkan.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vulkan.dispatchTable.allocateCommandBuffers(&allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vulkan.dispatchTable.beginCommandBuffer(commandBuffer, &beginInfo);

    return new GpuCommandBuffer_T { commandBuffer };
}

void gpuSubmit(GpuQueue queue, Span<GpuCommandBuffer> commandBuffers, GpuSemaphore semaphore, uint64_t value)
{
    std::vector<VkCommandBuffer> vkCommandBuffers;
    for (auto cb : commandBuffers)
    {
        vkCommandBuffers.push_back(cb->commandBuffer);
        vulkan.dispatchTable.endCommandBuffer(cb->commandBuffer);
    }

    VkTimelineSemaphoreSubmitInfo timelineInfo = {};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &value;

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.commandBufferCount = static_cast<uint32_t>(vkCommandBuffers.size());
    submitInfo.pCommandBuffers = vkCommandBuffers.data();
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &semaphore->semaphore;

    vulkan.dispatchTable.queueSubmit(queue->queue, 1, &submitInfo, VK_NULL_HANDLE);
    
    vulkan.submittedCommandBuffers[{ semaphore->semaphore, value }] = vkCommandBuffers;

    for (auto cb : commandBuffers)
    {
        delete cb; // Frees the wrapper, but not the actual VkCommandBuffer
    }
}

GpuSemaphore gpuCreateSemaphore(uint64_t initValue)
{
    VkSemaphoreTypeCreateInfo semaphoreTypeInfo = {};
    semaphoreTypeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semaphoreTypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphoreTypeInfo.initialValue = initValue;

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &semaphoreTypeInfo;

    VkSemaphore semaphore;
    vulkan.dispatchTable.createSemaphore(&semaphoreInfo, nullptr, &semaphore);

    return new GpuSemaphore_T { semaphore };
}

void gpuWaitSemaphore(GpuSemaphore sema, uint64_t value)
{
    VkSemaphoreWaitInfo waitInfo = {};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &sema->semaphore;
    waitInfo.pValues = &value;

    vulkan.dispatchTable.waitSemaphores(&waitInfo, UINT64_MAX);

    // Free command buffers for this semaphore, and any earlier values if they exist
    for (size_t i = value; i > 0; i--)
    {
        if (vulkan.submittedCommandBuffers.find({sema->semaphore, i}) != vulkan.submittedCommandBuffers.end())
        {
            auto& cmdBuffers = vulkan.submittedCommandBuffers[{sema->semaphore, i}];
            vulkan.dispatchTable.freeCommandBuffers(vulkan.commandPool, static_cast<uint32_t>(cmdBuffers.size()), cmdBuffers.data());
            vulkan.submittedCommandBuffers.erase({sema->semaphore, i});
        }
        else
        {
            break;
        }
    }
}

void gpuDestroySemaphore(GpuSemaphore sema)
{
    vulkan.dispatchTable.destroySemaphore(sema->semaphore, nullptr);
    delete sema;
}

void gpuMemCpy(GpuCommandBuffer cb, void* destGpu, void* srcGpu, uint64_t size)
{
    Allocation src = vulkan.findAllocation(reinterpret_cast<VkDeviceAddress>(srcGpu));
    Allocation dst = vulkan.findAllocation(reinterpret_cast<VkDeviceAddress>(destGpu));

    if (src.buffer == VK_NULL_HANDLE || dst.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = reinterpret_cast<VkDeviceAddress>(srcGpu) - src.address;
    copyRegion.dstOffset = reinterpret_cast<VkDeviceAddress>(destGpu) - dst.address;
    copyRegion.size = size;

    vulkan.dispatchTable.cmdCopyBuffer(cb->commandBuffer, src.buffer, dst.buffer, 1, &copyRegion);
}

void gpuCopyToTexture(GpuCommandBuffer cb, void* destGpu, void* srcGpu, GpuTexture texture)
{
    Allocation src = vulkan.findAllocation(reinterpret_cast<VkDeviceAddress>(srcGpu));
    Allocation dst = vulkan.findAllocation(reinterpret_cast<VkDeviceAddress>(destGpu));

    if (src.buffer == VK_NULL_HANDLE || dst.image == VK_NULL_HANDLE)
    {
        return;
    }

    VkBufferImageCopy region = {};
    region.bufferOffset = reinterpret_cast<VkDeviceAddress>(srcGpu) - src.address;
    region.bufferRowLength = 0; 
    region.bufferImageHeight = 0; 
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { texture->desc.dimensions.x, texture->desc.dimensions.y, texture->desc.dimensions.z };

    vulkan.dispatchTable.cmdCopyBufferToImage(cb->commandBuffer, src.buffer, dst.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

void gpuCopyFromTexture(GpuCommandBuffer cb, void* destGpu, void* srcGpu, GpuTexture texture)
{
    Allocation src = vulkan.findAllocation(reinterpret_cast<VkDeviceAddress>(srcGpu));
    Allocation dst = vulkan.findAllocation(reinterpret_cast<VkDeviceAddress>(destGpu));

    if (dst.buffer == VK_NULL_HANDLE || src.image == VK_NULL_HANDLE)
    {
        return;
    }

    VkBufferImageCopy region = {};
    region.bufferOffset = reinterpret_cast<VkDeviceAddress>(destGpu) - dst.address;
    region.bufferRowLength = 0; 
    region.bufferImageHeight = 0; 
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { texture->desc.dimensions.x, texture->desc.dimensions.y, texture->desc.dimensions.z };

    vulkan.dispatchTable.cmdCopyImageToBuffer(cb->commandBuffer, src.image, VK_IMAGE_LAYOUT_GENERAL, dst.buffer, 1, &region);
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer cb, void *ptrGpu)
{
}

void gpuBarrier(GpuCommandBuffer cb, STAGE before, STAGE after, HAZARD_FLAGS hazards)
{
}

void gpuSignalAfter(GpuCommandBuffer cb, STAGE before, void *ptrGpu, uint64_t value, SIGNAL signal)
{
}

void gpuWaitBefore(GpuCommandBuffer cb, STAGE after, void *ptrGpu, uint64_t value, OP op, HAZARD_FLAGS hazards, uint64_t mask)
{
}

void gpuSetPipeline(GpuCommandBuffer cb, GpuPipeline pipeline)
{
    vulkan.dispatchTable.cmdBindPipeline(
        cb->commandBuffer, 
        pipeline->bindPoint,
        pipeline->pipeline
    );

    vulkan.currentPipeline = pipeline;
}

void gpuSetDepthStencilState(GpuCommandBuffer cb, GpuDepthStencilState state)
{
}

void gpuSetBlendState(GpuCommandBuffer cb, GpuBlendState state)
{
}

void gpuDispatch(GpuCommandBuffer cb, void* dataGpu, uint3 gridDimensions)
{
    Allocation alloc = vulkan.findAllocation(reinterpret_cast<VkDeviceAddress>(dataGpu));
    if (alloc.buffer != VK_NULL_HANDLE)
    {
        VkDescriptorBufferInfo bufferInfo = {};
        bufferInfo.buffer = alloc.buffer;
        bufferInfo.offset = reinterpret_cast<VkDeviceAddress>(dataGpu) - alloc.address;
        bufferInfo.range = alloc.size - bufferInfo.offset;

        VkWriteDescriptorSet writeDescriptorSet = {};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstBinding = 0;
        writeDescriptorSet.dstArrayElement = 0;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.pBufferInfo = &bufferInfo;
        
        vulkan.dispatchTable.cmdPushDescriptorSetKHR(
            cb->commandBuffer,
            vulkan.currentPipeline->bindPoint,
            vulkan.currentPipeline->layout,
            0,
            1,
            &writeDescriptorSet
        );
    }

    vulkan.dispatchTable.cmdDispatch(
        cb->commandBuffer, 
        gridDimensions.x, 
        gridDimensions.y, 
        gridDimensions.z
    );
}

void gpuDispatchIndirect(GpuCommandBuffer cb, void* dataGpu, void* gridDimensionsGpu)
{
}

void gpuBeginRenderPass(GpuCommandBuffer cb, GpuRenderPassDesc desc)
{
    VkRenderingInfo renderingInfo = {};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;

    // TODO: Fill out renderingInfo from desc (GpuRenderPassDesc is incomplete)

    vulkan.dispatchTable.cmdBeginRendering(cb->commandBuffer, nullptr);
}

void gpuEndRenderPass(GpuCommandBuffer cb)
{
    vulkan.dispatchTable.cmdEndRendering(cb->commandBuffer);
}

void gpuDrawIndexedInstanced(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, uint32_t indexCount, uint32_t instanceCount)
{
}

void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, void* argsGpu)
{
}

void gpuDrawIndexedInstancedIndirectMulti(GpuCommandBuffer cb, void* dataVxGpu, uint32_t vxStride, void* dataPxGpu, uint32_t pxStride, void* argsGpu, void* drawCountGpu)
{
}

void gpuDrawMeshlets(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, uint3 dim)
{
}

void gpuDrawMeshletsIndirect(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, void *dimGpu)
{
}
