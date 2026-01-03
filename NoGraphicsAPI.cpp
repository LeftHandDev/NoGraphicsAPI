#define GPU_EXPOSE_INTERNAL
#include "NoGraphicsAPI.h"
#include "External/VkBootstrap.h"

#include "Config.h"

#include <map>
#include <vector>

struct GpuPipeline_T 
{ 
    VkPipeline pipeline; 
    VkPipelineBindPoint bindPoint;
};
struct GpuTexture_T 
{ 
    GpuTextureDesc desc = {}; 
    VkImage image = VK_NULL_HANDLE; 
    VkImageView view = VK_NULL_HANDLE;
};
struct GpuDepthStencilState_T {  };
struct GpuBlendState_T { };
struct GpuQueue_T { VkQueue queue; };
struct GpuCommandBuffer_T { VkCommandBuffer commandBuffer; };
struct GpuSemaphore_T { VkSemaphore semaphore; };
#ifdef GPU_SURFACE_EXTENSION
struct GpuSurface_T { VkSurfaceKHR surface = VK_NULL_HANDLE; };
struct GpuSwapchain_T 
{ 
    VkSwapchainKHR swapchain = VK_NULL_HANDLE; 
    VkQueue presentQueue = VK_NULL_HANDLE;
    uint32_t imageIndex = 0; 
    GpuTextureDesc desc = {};
    std::vector<GpuTexture> images;
    std::vector<VkSemaphore> presentSemaphores;
};
#endif // GPU_SURFACE_EXTENSION
#ifdef GPU_RAY_TRACING_EXTENSION
struct GpuAccelerationStructure_T 
{ 
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    VkAccelerationStructureBuildRangeInfoKHR buildRange = {}; 
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
};
#endif // GPU_RAY_TRACING_EXTENSION

VkPipelineStageFlagBits gpuStageToVkStage(STAGE stage)
{
    switch (stage)
    {
    case STAGE_TRANSFER: return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case STAGE_COMPUTE: return VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    case STAGE_RASTER_COLOR_OUT: return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    case STAGE_PIXEL_SHADER: return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    case STAGE_VERTEX_SHADER: return VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
    case STAGE_ACCELERATION_STRUCTURE_BUILD: return VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    default: return VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
}

FORMAT gpuVkFormatToGpuFormat(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R8G8B8A8_UNORM: return FORMAT_RGBA8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return FORMAT_BGRA8_SRGB;
    case VK_FORMAT_D32_SFLOAT: return FORMAT_D32_FLOAT;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return FORMAT_RG11B10_FLOAT;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return FORMAT_RGB10_A2_UNORM;
    default: return FORMAT_NONE;
    }
}

VkFormat gpuFormatToVkFormat(FORMAT format)
{
    switch (format)
    {
    case FORMAT_RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case FORMAT_BGRA8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;
    case FORMAT_D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
    case FORMAT_RG11B10_FLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case FORMAT_RGB10_A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case FORMAT_RGB32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

USAGE_FLAGS gpuVkUsageToGpuUsage(VkImageUsageFlags usage)
{
    uint32_t result = 0;
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
    {
        result = static_cast<uint32_t>(USAGE_SAMPLED) | result;
    }
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
    {
        result = static_cast<uint32_t>(USAGE_STORAGE) | result;
    }
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    {
        result = static_cast<uint32_t>(USAGE_COLOR_ATTACHMENT) | result;
    }
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    {
        result = static_cast<uint32_t>(USAGE_DEPTH_STENCIL_ATTACHMENT) | result;
    }
    if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
    {
        result = static_cast<uint32_t>(USAGE_TRANSFER_DST) | result;
    }

    return static_cast<USAGE_FLAGS>(result);
}

VkImageUsageFlagBits gpuGpuUsageToVkUsage(USAGE_FLAGS usage)
{
    uint32_t result = 0;
    if (usage & USAGE_SAMPLED)
    {
        result = static_cast<uint32_t>(VK_IMAGE_USAGE_SAMPLED_BIT) | result;
    }
    if (usage & USAGE_STORAGE)
    {
        result = static_cast<uint32_t>(VK_IMAGE_USAGE_STORAGE_BIT) | result;
    }
    if (usage & USAGE_COLOR_ATTACHMENT)
    {
        result = static_cast<uint32_t>(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) | result;
    }
    if (usage & USAGE_DEPTH_STENCIL_ATTACHMENT)
    {
        result = static_cast<uint32_t>(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) | result;
    }
    if (usage & USAGE_TRANSFER_DST)
    {
        result = static_cast<uint32_t>(VK_IMAGE_USAGE_TRANSFER_DST_BIT) | result;
    }

    return static_cast<VkImageUsageFlagBits>(result);
}

struct Allocation
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
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
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::map<std::pair<VkSemaphore, uint64_t>, std::vector<VkCommandBuffer>> submittedCommandBuffers;
    VkSampler defaultSampler = VK_NULL_HANDLE;
    VkFence acquireFence = VK_NULL_HANDLE;
    std::map<VkPipelineBindPoint, VkPipelineLayout> layout;
    VkDescriptorSetLayout textureSetLayout;
    VkDescriptorSetLayout rwTextureSetLayout; 
    VkDescriptorSetLayout samplerSetLayout;

    // Vulkan structs
    VkPhysicalDeviceMemoryProperties memoryProperties = {};
    VkPhysicalDeviceProperties2 physicalDeviceProperties2 = {};
    VkPhysicalDeviceDescriptorBufferPropertiesEXT descriptorBufferProperties = {};

    // Allocation tracking
    std::vector<Allocation> allocations;
    Allocation samplerDescriptors;

    // Opaque handles
    GpuQueue graphicsQueue = nullptr;
    std::map<GpuCommandBuffer, GpuPipeline> currentPipeline;

    // Buffer descriptor sets
    VkDeviceSize descriptorSetLayoutSize = 0;
    VkDeviceSize descriptorSetLayoutOffset = 0;
    VkDeviceSize rwDescriptorSetLayoutSize = 0;
    VkDeviceSize rwDescriptorSetLayoutOffset = 0;
    VkDeviceSize samplerDescriptorSetLayoutSize = 0;
    VkDeviceSize samplerDescriptorSetLayoutOffset = 0;

    // Descriptors
    uint32_t descriptorCount = 1024;

    Vulkan()
    {
        std::vector<const char*> requiredInstanceExtensions = {};
        std::vector<const char*> requiredDeviceExtensions = {
            VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
            VK_EXT_MESH_SHADER_EXTENSION_NAME,
            VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME
        };

#ifdef GPU_SURFACE_EXTENSION
        requiredInstanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
        requiredInstanceExtensions.push_back("VK_KHR_win32_surface");
#else
        requiredInstanceExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
        requiredDeviceExtensions.push_back(VK_KHR_DISPLAY_SWAPCHAIN_EXTENSION_NAME);
#endif
        requiredDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#endif // GPU_SURFACE_EXTENSION
#ifdef GPU_RAY_TRACING_EXTENSION
        requiredDeviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        requiredDeviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        requiredDeviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
#endif // GPU_RAY_TRACING_EXTENSION

        vkb::InstanceBuilder instanceBuilder;

        auto instanceRet = instanceBuilder.request_validation_layers(true)
            .require_api_version(VK_API_VERSION_1_4)
            .enable_extensions(requiredInstanceExtensions)
            .build();
        instance = instanceRet.value();
        instanceDispatchTable = instance.make_table();

        vkb::PhysicalDeviceSelector deviceSelector{ instance };
        auto physicalDeviceRet = deviceSelector
            .add_required_extensions(requiredDeviceExtensions)
            .defer_surface_initialization()
            .select();
        physicalDevice = physicalDeviceRet.value();

#ifdef GPU_RAY_TRACING_EXTENSION
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {};
        rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        rayQueryFeatures.rayQuery = VK_TRUE;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {};
        accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        accelerationStructureFeatures.accelerationStructure = VK_TRUE;
#endif // GPU_RAY_TRACING_EXTENSION

        VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferFeatures = {};
        descriptorBufferFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
        descriptorBufferFeatures.descriptorBuffer = VK_TRUE;

        VkPhysicalDeviceVulkan13Features physicalDeviceVulkan13Features = {};
        physicalDeviceVulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        physicalDeviceVulkan13Features.synchronization2 = VK_TRUE;
        physicalDeviceVulkan13Features.dynamicRendering = VK_TRUE;

        VkPhysicalDeviceVulkan12Features physicalDeviceVulkan12Features = {};
        physicalDeviceVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        physicalDeviceVulkan12Features.timelineSemaphore = VK_TRUE;
        physicalDeviceVulkan12Features.bufferDeviceAddress = VK_TRUE;
        physicalDeviceVulkan12Features.runtimeDescriptorArray = VK_TRUE;

        physicalDevice.features.shaderInt64 = VK_TRUE;

        bool result = physicalDevice.enable_extension_features_if_present(physicalDeviceVulkan12Features);
        result = physicalDevice.enable_extension_features_if_present(physicalDeviceVulkan13Features) && result;
#ifdef GPU_RAY_TRACING_EXTENSION
        result = physicalDevice.enable_extension_features_if_present(rayQueryFeatures) && result;
        result = physicalDevice.enable_extension_features_if_present(accelerationStructureFeatures) && result;
#endif // GPU_RAY_TRACING_EXTENSION
        instanceDispatchTable.getPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        descriptorBufferProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;
        physicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        physicalDeviceProperties2.pNext = &descriptorBufferProperties;
        instanceDispatchTable.getPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties2);

        assert(descriptorBufferProperties.sampledImageDescriptorSize == sizeof(GpuTextureDescriptor));
        assert(descriptorBufferProperties.storageImageDescriptorSize == sizeof(GpuTextureDescriptor));

        vkb::DeviceBuilder deviceBuilder{ physicalDevice };
        auto deviceRet = deviceBuilder.add_pNext(&descriptorBufferFeatures).build();
        device = deviceRet.value();
        dispatchTable = device.make_table();

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = device.get_queue_index(vkb::QueueType::graphics).value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        dispatchTable.createCommandPool(&poolInfo, nullptr, &commandPool);

        graphicsQueue = new GpuQueue_T{ device.get_queue(vkb::QueueType::graphics).value() };

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        dispatchTable.createSampler(&samplerInfo, nullptr, &defaultSampler);

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        dispatchTable.createFence(&fenceInfo, nullptr, &acquireFence);

        createPipelineLayout();
    }

    ~Vulkan()
    {
        dispatchTable.deviceWaitIdle();
        dispatchTable.destroyCommandPool(commandPool, nullptr);
        dispatchTable.destroySampler(defaultSampler, nullptr);
        dispatchTable.destroyFence(acquireFence, nullptr);
        dispatchTable.destroyDescriptorSetLayout(textureSetLayout, nullptr);
        dispatchTable.destroyDescriptorSetLayout(rwTextureSetLayout, nullptr);
        dispatchTable.destroyDescriptorSetLayout(samplerSetLayout, nullptr);
        dispatchTable.destroyPipelineLayout(layout[VK_PIPELINE_BIND_POINT_GRAPHICS], nullptr);
        dispatchTable.destroyPipelineLayout(layout[VK_PIPELINE_BIND_POINT_COMPUTE], nullptr);
        // dispatchTable.destroyPipelineLayout(layout[VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR], nullptr);
        delete graphicsQueue;
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

    void freeAllocation(Allocation alloc)
    {
        if (alloc.ptr)
        {
            dispatchTable.unmapMemory(alloc.memory);
        }

        dispatchTable.destroyBuffer(alloc.buffer, nullptr);
        dispatchTable.freeMemory(alloc.memory, nullptr);

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
        imageInfo.format = gpuFormatToVkFormat(desc.format);
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = gpuGpuUsageToVkUsage(desc.usage);
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkMemoryRequirements imageMemoryRequirements = {};

        VkImage image;
        dispatchTable.createImage(&imageInfo, nullptr, &image);

        return image;
    }

    void createPipelineLayout()
    {
        VkDescriptorSetLayoutBinding textureBinding = {};
        textureBinding.binding = 0;
        textureBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        textureBinding.descriptorCount = descriptorCount;
        textureBinding.stageFlags = VK_SHADER_STAGE_ALL;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        descriptorSetLayoutCreateInfo.bindingCount = 1;
        descriptorSetLayoutCreateInfo.pBindings = &textureBinding;
        VkDescriptorSetLayout textureSetLayout;
        dispatchTable.createDescriptorSetLayout(&descriptorSetLayoutCreateInfo, nullptr, &textureSetLayout);

        VkDescriptorSetLayoutBinding rwTextureBinding = {};
        rwTextureBinding.binding = 0;
        rwTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        rwTextureBinding.descriptorCount = descriptorCount;
        rwTextureBinding.stageFlags = VK_SHADER_STAGE_ALL;

        VkDescriptorSetLayoutCreateInfo rwDescriptorSetLayoutCreateInfo = {};
        rwDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        rwDescriptorSetLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        rwDescriptorSetLayoutCreateInfo.bindingCount = 1;
        rwDescriptorSetLayoutCreateInfo.pBindings = &rwTextureBinding;
        VkDescriptorSetLayout rwTextureSetLayout;
        dispatchTable.createDescriptorSetLayout(&rwDescriptorSetLayoutCreateInfo, nullptr, &rwTextureSetLayout);

        VkDescriptorSetLayoutBinding samplerBinding = {};
        samplerBinding.binding = 0;
        samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        samplerBinding.descriptorCount = descriptorCount;
        samplerBinding.stageFlags = VK_SHADER_STAGE_ALL;

        VkDescriptorSetLayoutCreateInfo samplerDescriptorSetLayoutCreateInfo = {};
        samplerDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        samplerDescriptorSetLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        samplerDescriptorSetLayoutCreateInfo.bindingCount = 1;
        samplerDescriptorSetLayoutCreateInfo.pBindings = &samplerBinding;
        VkDescriptorSetLayout samplerSetLayout;
        dispatchTable.createDescriptorSetLayout(&samplerDescriptorSetLayoutCreateInfo, nullptr, &samplerSetLayout);

        dispatchTable.getDescriptorSetLayoutSizeEXT(textureSetLayout, &descriptorSetLayoutSize);
        dispatchTable.getDescriptorSetLayoutBindingOffsetEXT(textureSetLayout, 0, &descriptorSetLayoutOffset);
        dispatchTable.getDescriptorSetLayoutSizeEXT(rwTextureSetLayout, &rwDescriptorSetLayoutSize);
        dispatchTable.getDescriptorSetLayoutBindingOffsetEXT(rwTextureSetLayout, 0, &rwDescriptorSetLayoutOffset);
        dispatchTable.getDescriptorSetLayoutSizeEXT(samplerSetLayout, &samplerDescriptorSetLayoutSize);
        dispatchTable.getDescriptorSetLayoutBindingOffsetEXT(samplerSetLayout, 0, &samplerDescriptorSetLayoutOffset);

        // Graphics
        {
            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = 
                VK_SHADER_STAGE_VERTEX_BIT | 
                VK_SHADER_STAGE_TASK_BIT_EXT | 
                VK_SHADER_STAGE_MESH_BIT_EXT |
                VK_SHADER_STAGE_FRAGMENT_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(VkDeviceAddress) * 3; // vertex/mesh + pixel + indirect multi strides

            VkDescriptorSetLayout descriptorSetLayouts[] = { textureSetLayout, rwTextureSetLayout, samplerSetLayout };

            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
            pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutCreateInfo.setLayoutCount = 3;
            pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts;
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
            pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

            VkPipelineLayout pipelineLayout;
            dispatchTable.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

            layout[VK_PIPELINE_BIND_POINT_GRAPHICS] = pipelineLayout;
        }

        // Compute
        {
            VkPushConstantRange pushConstantRange = {};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(VkDeviceAddress);

            VkDescriptorSetLayout descriptorSetLayouts[] = { textureSetLayout, rwTextureSetLayout, samplerSetLayout };

            VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
            pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutCreateInfo.setLayoutCount = 3;
            pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts;
            pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
            pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

            VkPipelineLayout pipelineLayout;
            dispatchTable.createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

            layout[VK_PIPELINE_BIND_POINT_COMPUTE] = pipelineLayout;
        }

        // Ray tracing ignored for now
    }
};

Vulkan* vulkan = nullptr;

void* gpuVulkanInstance()
{
    return vulkan->instance.instance;
}

GpuSurface gpuCreateSurface(void *vulkanSurface)
{
    return new GpuSurface_T { static_cast<VkSurfaceKHR>(vulkanSurface) };
}

void *gpuVulkanSurface(GpuSurface surface)
{
    if (surface == nullptr)
    {
        return nullptr;
    }

    return surface->surface;
}

void gpuDestroySurface(GpuSurface surface)
{
    if (surface == nullptr)
    {
        return;
    }

    delete surface;
}

RESULT gpuCreateDevice()
{
    if (vulkan == nullptr)
    {
        vulkan = new Vulkan();
    }

    return RESULT_SUCCESS;
}

void gpuDestroyDevice()
{
    if (vulkan != nullptr)
    {
        delete vulkan;
        vulkan = nullptr;
    }
}

void* gpuMalloc(size_t bytes, MEMORY memory)
{
    return gpuMalloc(bytes, GPU_DEFAULT_ALIGNMENT, memory);
}

void* gpuMallocHidden(size_t bytes, size_t align, MEMORY memory, bool sampler = false)
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
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | 
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

            if (sampler)
            {
                usage |= VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
            }
            else
            {
                usage |= VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;
            }

        auto properties = 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        auto alloc = vulkan->createAllocation(bytes, usage, properties, align);
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
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;

        auto properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        auto alloc = vulkan->createAllocation(bytes, usage, properties, align);
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

        auto alloc = vulkan->createAllocation(bytes, usage, properties, align);
        return alloc.ptr;    
    }
    };

    return nullptr;
}

void* gpuMalloc(size_t bytes, size_t align, MEMORY memory)
{
    return gpuMallocHidden(bytes, align, memory);
}

void gpuFree(void *ptr)
{
    for (auto& alloc : vulkan->allocations)
    {
        if (alloc.ptr == ptr || reinterpret_cast<VkDeviceAddress>(ptr) == alloc.address)
        {
            vulkan->freeAllocation(alloc);
            return;
        }
    }
}

void* gpuHostToDevicePointer(void *ptr)
{
    Allocation alloc = vulkan->findAllocation(ptr);
    if (alloc.buffer != VK_NULL_HANDLE)
    {
        return reinterpret_cast<void*>(alloc.address + (static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(alloc.ptr)));
    }
    return nullptr;
}

GpuTextureSizeAlign gpuTextureSizeAlign(GpuTextureDesc desc)
{
    VkImage image = vulkan->createImage(desc);

    VkMemoryRequirements imageMemoryRequirements = {};
    vulkan->dispatchTable.getImageMemoryRequirements(image, &imageMemoryRequirements);
    vulkan->dispatchTable.destroyImage(image, nullptr);

    return { imageMemoryRequirements.size, imageMemoryRequirements.alignment };
}

GpuTexture gpuCreateTexture(GpuTextureDesc desc, void* ptrGpu)
{
    Allocation alloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(ptrGpu));

    if (alloc.buffer == VK_NULL_HANDLE)
    {
        return nullptr;
    }

    VkImage image = vulkan->createImage(desc);

    VkDeviceSize offset = reinterpret_cast<VkDeviceAddress>(ptrGpu) - alloc.address;
    vulkan->dispatchTable.bindImageMemory(image, alloc.memory, offset);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = desc.type == TEXTURE_1D ? VK_IMAGE_VIEW_TYPE_1D :
                        desc.type == TEXTURE_2D ? VK_IMAGE_VIEW_TYPE_2D :
                        desc.type == TEXTURE_3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_MAX_ENUM;
    viewInfo.format = gpuFormatToVkFormat(desc.format);
    viewInfo.subresourceRange.aspectMask = desc.usage == USAGE_DEPTH_STENCIL_ATTACHMENT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = desc.mipCount;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = desc.layerCount;

    VkImageView imageView = VK_NULL_HANDLE;
    vulkan->dispatchTable.createImageView(&viewInfo, nullptr, &imageView);

    return new GpuTexture_T { desc, image, imageView};
}

void gpuDestroyTexture(GpuTexture texture)
{
    if (texture == nullptr)
    {
        return;
    }

    vulkan->dispatchTable.destroyImageView(texture->view, nullptr);
    vulkan->dispatchTable.destroyImage(texture->image, nullptr);
    delete texture;
}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc)
{
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = vulkan->defaultSampler;
    imageInfo.imageView = texture->view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorGetInfoEXT descriptorGetInfo = {};
    descriptorGetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    descriptorGetInfo.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptorGetInfo.data.pSampledImage = &imageInfo;

    GpuTextureDescriptor descriptor = {};
    vulkan->dispatchTable.getDescriptorEXT(&descriptorGetInfo, vulkan->descriptorBufferProperties.sampledImageDescriptorSize, descriptor.data);
    return descriptor;
}

GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc)
{
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = VK_NULL_HANDLE;
    imageInfo.imageView = texture->view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorGetInfoEXT descriptorGetInfo = {};
    descriptorGetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    descriptorGetInfo.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorGetInfo.data.pStorageImage = &imageInfo;

    GpuTextureDescriptor descriptor = {};
    vulkan->dispatchTable.getDescriptorEXT(&descriptorGetInfo, vulkan->descriptorBufferProperties.storageImageDescriptorSize, descriptor.data);
    return descriptor;
}

GpuPipeline gpuCreateComputePipeline(ByteSpan computeIR)
{
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = computeIR.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(computeIR.data());
    VkShaderModule shaderModule;
    vulkan->dispatchTable.createShaderModule(&shaderModuleCreateInfo, nullptr, &shaderModule);

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.layout = vulkan->layout[VK_PIPELINE_BIND_POINT_COMPUTE];
    pipelineCreateInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pName = "main";

    VkPipeline pipeline;
    vulkan->dispatchTable.createComputePipelines(VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);
    vulkan->dispatchTable.destroyShaderModule(shaderModule, nullptr);

    return new GpuPipeline_T { pipeline, VK_PIPELINE_BIND_POINT_COMPUTE };
}

VkPipeline gpuCreateGraphicsPipeline(ByteSpan vertexIR, ByteSpan meshletIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    bool vertex = vertexIR.size() > 0;
    ByteSpan actualIR = vertex ? vertexIR : meshletIR;

    VkShaderModuleCreateInfo vertexShaderModuleCreateInfo = {};
    vertexShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vertexShaderModuleCreateInfo.codeSize = actualIR.size();
    vertexShaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(actualIR.data());
    VkShaderModule vertexShaderModule;
    vulkan->dispatchTable.createShaderModule(&vertexShaderModuleCreateInfo, nullptr, &vertexShaderModule);

    VkShaderModuleCreateInfo pixelShaderModuleCreateInfo = {};
    pixelShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    pixelShaderModuleCreateInfo.codeSize = pixelIR.size();
    pixelShaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(pixelIR.data());
    VkShaderModule pixelShaderModule;
    vulkan->dispatchTable.createShaderModule(&pixelShaderModuleCreateInfo, nullptr, &pixelShaderModule);

    VkPipelineShaderStageCreateInfo shaderStages[2] = {};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = vertex ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_MESH_BIT_NV;
    shaderStages[0].module = vertexShaderModule;
    shaderStages[0].pName = "main";

    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = pixelShaderModule;
    shaderStages[1].pName = "main";

    std::vector<VkFormat> colorFormats;
    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;
    
    for (auto& target : desc.colorTargets)
    {
        colorFormats.push_back(gpuFormatToVkFormat(target.format));

        VkPipelineColorBlendAttachmentState blendAttachment = {};

        if (desc.blendState)
        {
            GpuBlendDesc blendDesc = *desc.blendState;
            blendAttachment.blendEnable = VK_TRUE;
            // TODO: blend state
        }
        else
        {
            blendAttachment.blendEnable = VK_FALSE;
            blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
            blendAttachment.colorWriteMask = 
                VK_COLOR_COMPONENT_R_BIT | 
                VK_COLOR_COMPONENT_G_BIT | 
                VK_COLOR_COMPONENT_B_BIT | 
                VK_COLOR_COMPONENT_A_BIT;
        }

        blendAttachments.push_back(blendAttachment);
    }

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = desc.colorTargets.size();
    pipelineRenderingInfo.pColorAttachmentFormats = colorFormats.data();

    VkPipelineColorBlendStateCreateInfo blendState = {};
    blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendState.attachmentCount = blendAttachments.size();
    blendState.pAttachments = blendAttachments.data();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {};
    inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyInfo.topology = desc.topology == TOPOLOGY_TRIANGLE_LIST ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST :
                                 desc.topology == TOPOLOGY_TRIANGLE_STRIP ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP :
                                 desc.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN : VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    viewportState.pViewports = nullptr;
    viewportState.pScissors = nullptr;

    VkPipelineMultisampleStateCreateInfo multisampleState = {};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.rasterizationSamples = static_cast<VkSampleCountFlagBits>(desc.sampleCount);
    multisampleState.alphaToCoverageEnable = desc.alphaToCoverage ? VK_TRUE : VK_FALSE;

    VkPipelineRasterizationStateCreateInfo rasterizationState = {};
    rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationState.cullMode = desc.cull != CULL_NONE ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    rasterizationState.frontFace = desc.cull == CULL_CW ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationState.lineWidth = 1.0f;
    rasterizationState.depthClampEnable = VK_FALSE;
    rasterizationState.rasterizerDiscardEnable = VK_FALSE;  
    rasterizationState.depthBiasEnable = VK_FALSE;

    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = &pipelineRenderingInfo;
    pipelineCreateInfo.layout = vulkan->layout[VK_PIPELINE_BIND_POINT_GRAPHICS];
    pipelineCreateInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyInfo;
    pipelineCreateInfo.pColorBlendState = &blendState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = 2;

    VkPipeline pipeline;
    vulkan->dispatchTable.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);
    vulkan->dispatchTable.destroyShaderModule(vertexShaderModule, nullptr);
    vulkan->dispatchTable.destroyShaderModule(pixelShaderModule, nullptr);

    return pipeline;
}

GpuPipeline gpuCreateGraphicsPipeline(ByteSpan vertexIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    VkPipeline pipeline = gpuCreateGraphicsPipeline(vertexIR, ByteSpan{}, pixelIR, desc);
    return new GpuPipeline_T { pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS };
}

GpuPipeline gpuCreateGraphicsMeshletPipeline(ByteSpan meshletIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    VkPipeline pipeline = gpuCreateGraphicsPipeline(ByteSpan{}, meshletIR, pixelIR, desc);
    return new GpuPipeline_T { pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS };
}

void gpuFreePipeline(GpuPipeline pipeline)
{
    vulkan->dispatchTable.destroyPipeline(pipeline->pipeline, nullptr);
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
    return vulkan->graphicsQueue;
}

GpuCommandBuffer gpuStartCommandRecording(GpuQueue queue)
{
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vulkan->commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vulkan->dispatchTable.allocateCommandBuffers(&allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vulkan->dispatchTable.beginCommandBuffer(commandBuffer, &beginInfo);

    return new GpuCommandBuffer_T { commandBuffer };
}

void gpuSubmit(GpuQueue queue, Span<GpuCommandBuffer> commandBuffers, GpuSemaphore semaphore, uint64_t value)
{
    std::vector<VkCommandBuffer> vkCommandBuffers;
    for (auto cb : commandBuffers)
    {
        vkCommandBuffers.push_back(cb->commandBuffer);
        vulkan->dispatchTable.endCommandBuffer(cb->commandBuffer);
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

    vulkan->dispatchTable.queueSubmit(queue->queue, 1, &submitInfo, VK_NULL_HANDLE);

    vulkan->submittedCommandBuffers[{ semaphore->semaphore, value }] = vkCommandBuffers;

    for (auto cb : commandBuffers)
    {
        vulkan->currentPipeline.erase(cb);
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
    vulkan->dispatchTable.createSemaphore(&semaphoreInfo, nullptr, &semaphore);

    return new GpuSemaphore_T { semaphore };
}

void gpuWaitSemaphore(GpuSemaphore sema, uint64_t value, uint64_t timeout)
{
    VkSemaphoreWaitInfo waitInfo = {};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &sema->semaphore;
    waitInfo.pValues = &value;

    vulkan->dispatchTable.waitSemaphores(&waitInfo, timeout);

    // Free command buffers for this semaphore, and any earlier values if they exist
    for (size_t i = value; i > 0; i--)
    {
        if (vulkan->submittedCommandBuffers.find({sema->semaphore, i}) != vulkan->submittedCommandBuffers.end())
        {
            auto& cmdBuffers = vulkan->submittedCommandBuffers[{sema->semaphore, i}];
            vulkan->dispatchTable.freeCommandBuffers(vulkan->commandPool, static_cast<uint32_t>(cmdBuffers.size()), cmdBuffers.data());
            vulkan->submittedCommandBuffers.erase({sema->semaphore, i});
        }
        else
        {
            break;
        }
    }
}

void gpuDestroySemaphore(GpuSemaphore sema)
{
    vulkan->dispatchTable.destroySemaphore(sema->semaphore, nullptr);
    delete sema;
}

void gpuMemCpy(GpuCommandBuffer cb, void* destGpu, void* srcGpu, uint64_t size)
{
    Allocation src = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(srcGpu));
    Allocation dst = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(destGpu));

    if (src.buffer == VK_NULL_HANDLE || dst.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = reinterpret_cast<VkDeviceAddress>(srcGpu) - src.address;
    copyRegion.dstOffset = reinterpret_cast<VkDeviceAddress>(destGpu) - dst.address;
    copyRegion.size = size;

    vulkan->dispatchTable.cmdCopyBuffer(cb->commandBuffer, src.buffer, dst.buffer, 1, &copyRegion);
}

void gpuCopyToTexture(GpuCommandBuffer cb, void* srcGpu, GpuTexture texture)
{
    Allocation src = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(srcGpu));

    if (src.buffer == VK_NULL_HANDLE || texture == nullptr)
    {
        return;
    }

    // VkImageMemoryBarrier barrier = {};
    // barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    // barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // barrier.image = texture->image;
    // barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // barrier.subresourceRange.baseMipLevel = 0;
    // barrier.subresourceRange.levelCount = 1;
    // barrier.subresourceRange.baseArrayLayer = 0;
    // barrier.subresourceRange.layerCount = 1;
    // barrier.srcAccessMask = VK_ACCESS_NONE;
    // barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    // vulkan->dispatchTable.cmdPipelineBarrier(
    //     cb->commandBuffer,
    //     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    //     VK_PIPELINE_STAGE_TRANSFER_BIT,
    //     0,
    //     0, nullptr,
    //     0, nullptr,
    //     1, &barrier
    // );

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

    vulkan->dispatchTable.cmdCopyBufferToImage(cb->commandBuffer, src.buffer, texture->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    // barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    // barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    // barrier.dstAccessMask = VK_ACCESS_NONE;

    // vulkan->dispatchTable.cmdPipelineBarrier(
    //     cb->commandBuffer,
    //     VK_PIPELINE_STAGE_TRANSFER_BIT,
    //     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    //     0,
    //     0, nullptr,
    //     0, nullptr,
    //     1, &barrier
    // );    
}

void gpuCopyFromTexture(GpuCommandBuffer cb, void* destGpu, GpuTexture texture)
{
    Allocation dst = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(destGpu));

    if (dst.buffer == VK_NULL_HANDLE || texture == nullptr)
    {
        return;
    }

    // VkImageMemoryBarrier barrier = {};
    // barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    // barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    // barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // barrier.image = texture->image;
    // barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // barrier.subresourceRange.baseMipLevel = 0;
    // barrier.subresourceRange.levelCount = 1;
    // barrier.subresourceRange.baseArrayLayer = 0;
    // barrier.subresourceRange.layerCount = 1;
    // barrier.srcAccessMask = VK_ACCESS_NONE;
    // barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    
    // vulkan->dispatchTable.cmdPipelineBarrier(
    //     cb->commandBuffer,
    //     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    //     VK_PIPELINE_STAGE_TRANSFER_BIT,
    //     0,
    //     0, nullptr,
    //     0, nullptr,
    //     1, &barrier
    // );

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

    vulkan->dispatchTable.cmdCopyImageToBuffer(cb->commandBuffer, texture->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buffer, 1, &region);

    // barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    // barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    // barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    // barrier.dstAccessMask = VK_ACCESS_NONE;

    // vulkan->dispatchTable.cmdPipelineBarrier(
    //     cb->commandBuffer,
    //     VK_PIPELINE_STAGE_TRANSFER_BIT,
    //     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    //     0,
    //     0, nullptr,
    //     0, nullptr,
    //     1, &barrier
    // );
}

void gpuBlitTexture(GpuCommandBuffer cb, GpuTexture destTexture, GpuTexture srcTexture)
{
    if (destTexture == nullptr || srcTexture == nullptr)
    {
        return;
    }

    // VkImageMemoryBarrier barriers[2] = {};
    // barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // barriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    // barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    // barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // barriers[0].image = srcTexture->image;
    // barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // barriers[0].subresourceRange.baseMipLevel = 0;
    // barriers[0].subresourceRange.levelCount = 1;
    // barriers[0].subresourceRange.baseArrayLayer = 0;
    // barriers[0].subresourceRange.layerCount = 1;
    // barriers[0].srcAccessMask = VK_ACCESS_NONE;
    // barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    // barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // barriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    // barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    // barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // barriers[1].image = destTexture->image;
    // barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // barriers[1].subresourceRange.baseMipLevel = 0;
    // barriers[1].subresourceRange.levelCount = 1;
    // barriers[1].subresourceRange.baseArrayLayer = 0;
    // barriers[1].subresourceRange.layerCount = 1;
    // barriers[1].srcAccessMask = VK_ACCESS_NONE;
    // barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    // vulkan->dispatchTable.cmdPipelineBarrier(
    //     cb->commandBuffer,
    //     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    //     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    //     0,
    //     0, nullptr,
    //     0, nullptr,
    //     2, barriers
    // );

    VkImageBlit blit = {};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = 0;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0] = { 0, 0, 0 };
    blit.srcOffsets[1] = { static_cast<int32_t>(srcTexture->desc.dimensions.x), static_cast<int32_t>(srcTexture->desc.dimensions.y), static_cast<int32_t>(srcTexture->desc.dimensions.z) };
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = 0;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0] = { 0, 0, 0 };
    blit.dstOffsets[1] = { static_cast<int32_t>(destTexture->desc.dimensions.x), static_cast<int32_t>(destTexture->desc.dimensions.y), static_cast<int32_t>(destTexture->desc.dimensions.z) };

    vulkan->dispatchTable.cmdBlitImage(
        cb->commandBuffer,
        srcTexture->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        destTexture->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &blit,
        VK_FILTER_NEAREST
    );

    // barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    // barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    // barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    // barriers[0].dstAccessMask = VK_ACCESS_NONE;
    // barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    // barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    // barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    // barriers[1].dstAccessMask = VK_ACCESS_NONE;

    // vulkan->dispatchTable.cmdPipelineBarrier(
    //     cb->commandBuffer,
    //     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    //     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    //     0,
    //     0, nullptr,
    //     0, nullptr,
    //     2, barriers
    // );
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer cb, void *ptrGpu)
{
    auto alloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(ptrGpu));
    if (alloc.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    if (vulkan->samplerDescriptors.buffer == VK_NULL_HANDLE)
    {
        auto cpu = gpuMallocHidden(
            vulkan->samplerDescriptorSetLayoutSize, 
            vulkan->descriptorBufferProperties.descriptorBufferOffsetAlignment, 
            MEMORY_DEFAULT, 
            true // sampler
        );

        vulkan->samplerDescriptors = vulkan->findAllocation(cpu);

        VkDescriptorGetInfoEXT samplerDescriptorGetInfo = {};
        samplerDescriptorGetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
        samplerDescriptorGetInfo.type = VK_DESCRIPTOR_TYPE_SAMPLER;
        samplerDescriptorGetInfo.data.pSampler = &vulkan->defaultSampler;
        vulkan->dispatchTable.getDescriptorEXT(&samplerDescriptorGetInfo, vulkan->descriptorBufferProperties.samplerDescriptorSize, vulkan->samplerDescriptors.ptr);
    }

    VkDescriptorBufferBindingInfoEXT bufferBindingInfo[2] = {};
    bufferBindingInfo[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    bufferBindingInfo[0].address = reinterpret_cast<VkDeviceAddress>(ptrGpu);
    bufferBindingInfo[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;

    bufferBindingInfo[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    bufferBindingInfo[1].address = vulkan->samplerDescriptors.address;
    bufferBindingInfo[1].usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;

    vulkan->dispatchTable.cmdBindDescriptorBuffersEXT(
        cb->commandBuffer,
        2,
        bufferBindingInfo
    );

    uint32_t indices[3] = { 0, 0, 1 }; // read, read/write, sampler
    VkDeviceSize offsets[3] = { 0, 0, 0 };

    vulkan->dispatchTable.cmdSetDescriptorBufferOffsetsEXT(
        cb->commandBuffer,
        vulkan->currentPipeline[cb]->bindPoint,
        vulkan->layout[vulkan->currentPipeline[cb]->bindPoint],
        0,
        3,
        indices,
        offsets
    );
}

void gpuBarrier(GpuCommandBuffer cb, STAGE before, STAGE after, HAZARD_FLAGS hazards)
{
    std::vector<VkMemoryBarrier> memoryBarriers;

    if (hazards & HAZARD_DRAW_ARGUMENTS)
    {
        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        memoryBarriers.push_back(memoryBarrier);
    }

    if (hazards & HAZARD_DESCRIPTORS)
    {
        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        memoryBarriers.push_back(memoryBarrier);
    }

    if (hazards & HAZARD_ACCELERATION_STRUCTURE)
    {
        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        memoryBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        memoryBarriers.push_back(memoryBarrier);
    }

    vulkan->dispatchTable.cmdPipelineBarrier(
        cb->commandBuffer,
        gpuStageToVkStage(before),
        gpuStageToVkStage(after),
        0,
        static_cast<uint32_t>(memoryBarriers.size()), memoryBarriers.data(),
        0, nullptr,
        0, nullptr
    );
}

void gpuSignalAfter(GpuCommandBuffer cb, STAGE before, void *ptrGpu, uint64_t value, SIGNAL signal)
{
    // TODO: implement
}

void gpuWaitBefore(GpuCommandBuffer cb, STAGE after, void *ptrGpu, uint64_t value, OP op, HAZARD_FLAGS hazards, uint64_t mask)
{
    // TODO: implement
}

void gpuSetPipeline(GpuCommandBuffer cb, GpuPipeline pipeline)
{
    vulkan->dispatchTable.cmdBindPipeline(
        cb->commandBuffer, 
        pipeline->bindPoint,
        pipeline->pipeline
    );

    vulkan->currentPipeline[cb] = pipeline;
}

void gpuSetDepthStencilState(GpuCommandBuffer cb, GpuDepthStencilState state)
{
    // TODO: implement
}

void gpuSetBlendState(GpuCommandBuffer cb, GpuBlendState state)
{
    // TODO: implement
}

void gpuDispatch(GpuCommandBuffer cb, void* dataGpu, uint3 gridDimensions)
{
    VkDeviceAddress address = reinterpret_cast<VkDeviceAddress>(dataGpu);
    vulkan->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
        vulkan->layout[vulkan->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(VkDeviceAddress),
        &address
    );

    vulkan->dispatchTable.cmdDispatch(
        cb->commandBuffer, 
        gridDimensions.x, 
        gridDimensions.y, 
        gridDimensions.z
    );
}

void gpuDispatchIndirect(GpuCommandBuffer cb, void* dataGpu, void* gridDimensionsGpu)
{
    VkDeviceAddress address = reinterpret_cast<VkDeviceAddress>(dataGpu);
    vulkan->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
        vulkan->layout[vulkan->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(VkDeviceAddress),
        &address
    );

    Allocation grid = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(gridDimensionsGpu));
    if (grid.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    vulkan->dispatchTable.cmdDispatchIndirect(
        cb->commandBuffer, 
        grid.buffer, 
        reinterpret_cast<VkDeviceAddress>(gridDimensionsGpu) - grid.address
    );
}

void gpuBeginRenderPass(GpuCommandBuffer cb, GpuRenderPassDesc desc)
{
    VkRenderingInfo renderingInfo = {};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = { 0, 0 };

    std::vector<VkRenderingAttachmentInfo> colorAttachments;
    for (const auto& colorTarget : desc.colorTargets)
    {
        VkRenderingAttachmentInfo colorAttachment = {};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = colorTarget->view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorAttachments.push_back(colorAttachment);
    }

    auto& colorTarget = desc.colorTargets[0];
    renderingInfo.renderArea.extent = { colorTarget->desc.dimensions.x, colorTarget->desc.dimensions.y };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = desc.colorTargets.size();
    renderingInfo.pColorAttachments = colorAttachments.data();

    vulkan->dispatchTable.cmdBeginRendering(cb->commandBuffer, &renderingInfo);

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(colorTarget->desc.dimensions.x);
    viewport.height = static_cast<float>(colorTarget->desc.dimensions.y);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vulkan->dispatchTable.cmdSetViewport(cb->commandBuffer, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = { colorTarget->desc.dimensions.x, colorTarget->desc.dimensions.y };
    vulkan->dispatchTable.cmdSetScissor(cb->commandBuffer, 0, 1, &scissor);
}

void gpuEndRenderPass(GpuCommandBuffer cb)
{
    vulkan->dispatchTable.cmdEndRendering(cb->commandBuffer);
}

void gpuDrawIndexedInstanced(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, uint32_t indexCount, uint32_t instanceCount)
{
    VkDeviceAddress pushConstants[3] = {
        reinterpret_cast<VkDeviceAddress>(vertexDataGpu),
        reinterpret_cast<VkDeviceAddress>(pixelDataGpu),
        0 // unused
    };

    vulkan->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
       vulkan->layout[vulkan->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 2,
        pushConstants
    );

    Allocation indexAlloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(indicesGpu));
    if (indexAlloc.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    vulkan->dispatchTable.cmdBindIndexBuffer(
        cb->commandBuffer,
        indexAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(indicesGpu) - indexAlloc.address,
        VK_INDEX_TYPE_UINT32
    );

    vulkan->dispatchTable.cmdDrawIndexed(
        cb->commandBuffer,
        indexCount,
        instanceCount,
        0,
        0,
        0
    );
}

void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, void* argsGpu)
{
    VkDeviceAddress pushConstants[3] = {
        reinterpret_cast<VkDeviceAddress>(vertexDataGpu),
        reinterpret_cast<VkDeviceAddress>(pixelDataGpu),
        0 // unused
    };

    vulkan->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
       vulkan->layout[vulkan->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 2,
        pushConstants
    );

    Allocation indexAlloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(indicesGpu));
    if (indexAlloc.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    vulkan->dispatchTable.cmdBindIndexBuffer(
        cb->commandBuffer,
        indexAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(indicesGpu) - indexAlloc.address,
        VK_INDEX_TYPE_UINT32
    );

    Allocation argsAlloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(argsGpu));
    if (argsAlloc.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    vulkan->dispatchTable.cmdDrawIndexedIndirect(
        cb->commandBuffer,
        argsAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(argsGpu) - argsAlloc.address,
        1,
        0
    );
}

void gpuDrawIndexedInstancedIndirectMulti(
    GpuCommandBuffer cb, 
    void* dataVxGpu, 
    uint32_t vxStride, 
    void* dataPxGpu, 
    uint32_t pxStride,
    void* indicesGpu, 
    void* argsGpu, 
    void* drawCountGpu)
{
    VkDeviceAddress pushConstants[3] = {
        reinterpret_cast<VkDeviceAddress>(dataVxGpu),
        reinterpret_cast<VkDeviceAddress>(dataPxGpu),
        static_cast<VkDeviceAddress>(vxStride | (static_cast<uint64_t>(pxStride) << 32)) // pack strides into a single 64-bit value
    };

    vulkan->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
        vulkan->layout[vulkan->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 3,
        pushConstants
    );

    Allocation indexAlloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(indicesGpu));
    if (indexAlloc.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    vulkan->dispatchTable.cmdBindIndexBuffer(
        cb->commandBuffer,
        indexAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(indicesGpu) - indexAlloc.address,
        VK_INDEX_TYPE_UINT32
    );

    Allocation argsAlloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(argsGpu));
    if (argsAlloc.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    Allocation countAlloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(drawCountGpu));
    if (countAlloc.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    VkDrawIndexedIndirectCommand drawCommand = {};

    vulkan->dispatchTable.cmdDrawIndexedIndirectCount(
        cb->commandBuffer,
        argsAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(argsGpu) - argsAlloc.address,
        countAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(drawCountGpu) - countAlloc.address,
        vulkan->physicalDeviceProperties2.properties.limits.maxDrawIndirectCount,
        sizeof(VkDrawIndexedIndirectCommand)
    );
}

void gpuDrawMeshlets(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, uint3 dim)
{
    VkDeviceAddress pushConstants[2] = {
        reinterpret_cast<VkDeviceAddress>(meshletDataGpu),
        reinterpret_cast<VkDeviceAddress>(pixelDataGpu)
    };

    vulkan->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
       vulkan->layout[vulkan->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 2,
        pushConstants
    );
    
    vulkan->dispatchTable.cmdDrawMeshTasksEXT(
        cb->commandBuffer,
        dim.x,
        dim.y,
        dim.z
    );
}

void gpuDrawMeshletsIndirect(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, void *dimGpu)
{
    VkDeviceAddress pushConstants[2] = {
        reinterpret_cast<VkDeviceAddress>(meshletDataGpu),
        reinterpret_cast<VkDeviceAddress>(pixelDataGpu)
    };

    vulkan->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
       vulkan->layout[vulkan->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 2,
        pushConstants
    );

    Allocation dimAlloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(dimGpu));
    if (dimAlloc.buffer == VK_NULL_HANDLE)
    {
        return;
    }

    vulkan->dispatchTable.cmdDrawMeshTasksIndirectEXT(
        cb->commandBuffer,
        dimAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(dimGpu) - dimAlloc.address,
        1,
        0
    );
}

#ifdef GPU_SURFACE_EXTENSION
GpuSwapchain gpuCreateSwapchain(GpuSurface surface, uint32_t images)
{
    struct GpuSurfaceImpl
    {
        VkSurfaceKHR surface;
    };
    auto surf = reinterpret_cast<GpuSurfaceImpl*>(surface);

    VkSurfaceFormatKHR surfaceFormat;

    auto builder = vkb::SwapchainBuilder{vulkan->device, surf->surface}
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    vulkan->device.surface = surf->surface;
    auto presentQueue = new GpuQueue_T{ vulkan->device.get_queue(vkb::QueueType::present).value() };
    auto swapchain = builder.build().value();

    auto desc = GpuTextureDesc {};
    desc.dimensions = uint3 { swapchain.extent.width, swapchain.extent.height, 1 };
    desc.format = gpuVkFormatToGpuFormat(swapchain.image_format);
    desc.usage = gpuVkUsageToGpuUsage(swapchain.image_usage_flags);

    std::vector<GpuTexture> swapchainImages;
    
    for (const auto& image : swapchain.get_images().value())
    {
        VkImageViewCreateInfo viewInfo = {};    
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchain.image_format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView view;
        vulkan->dispatchTable.createImageView(&viewInfo, nullptr, &view);

        swapchainImages.push_back(new GpuTexture_T {
            desc,
            image,
            view
        });
    }

    std::vector<VkSemaphore> presentSemaphores;

    for (size_t i = 0; i < swapchainImages.size(); i++)
    {
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore presentSemaphore;
        vulkan->dispatchTable.createSemaphore(&semaphoreInfo, nullptr, &presentSemaphore);
        presentSemaphores.push_back(presentSemaphore);
    }

    return new GpuSwapchain_T { 
        swapchain.swapchain, 
        presentQueue->queue, 
        0, 
        desc,
        swapchainImages, 
        presentSemaphores
    };
}

void gpuDestroySwapchain(GpuSwapchain swapchain)
{
    for (auto image : swapchain->images)
    {
        vulkan->dispatchTable.destroyImageView(image->view, nullptr);
        delete image;
    }
    vulkan->dispatchTable.destroySwapchainKHR(swapchain->swapchain, nullptr);
    for (auto sema : swapchain->presentSemaphores)
    {
        vulkan->dispatchTable.destroySemaphore(sema, nullptr);
    }
    delete swapchain;
}

GpuTextureDesc gpuSwapchainDesc(GpuSwapchain swapchain)
{
    return swapchain->desc;
}

GpuTexture gpuSwapchainImage(GpuSwapchain swapchain)
{
    vulkan->dispatchTable.resetFences(1, &vulkan->acquireFence);

    vulkan->dispatchTable.acquireNextImageKHR(
        swapchain->swapchain,
        UINT64_MAX,
        VK_NULL_HANDLE,
        vulkan->acquireFence,
        &swapchain->imageIndex
    );

    vulkan->dispatchTable.waitForFences(1, &vulkan->acquireFence, VK_TRUE, UINT64_MAX);

    return swapchain->images[swapchain->imageIndex];
}

void gpuPresent(GpuSwapchain swapchain, GpuSemaphore sema, uint64_t value)
{
    VkSemaphoreSubmitInfo waitInfo = {};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitInfo.semaphore = sema->semaphore;
    waitInfo.value = value;

    VkSemaphoreSubmitInfo signalInfo = {};
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalInfo.semaphore = swapchain->presentSemaphores[swapchain->imageIndex];
    signalInfo.value = 0;

    VkSubmitInfo2 submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalInfo;
    submitInfo.commandBufferInfoCount = 0;
    submitInfo.pCommandBufferInfos = nullptr;

    // Converts the timeline semaphore to a binary semaphore so we can present
    vulkan->dispatchTable.queueSubmit2(
        swapchain->presentQueue,
        1,
        &submitInfo,
        VK_NULL_HANDLE
    );

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain->swapchain;
    presentInfo.pImageIndices = &swapchain->imageIndex;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &swapchain->presentSemaphores[swapchain->imageIndex];

    vulkan->dispatchTable.queuePresentKHR(
        swapchain->presentQueue,
        &presentInfo
    );
}
#endif // GPU_SURFACE_EXTENSION

#ifdef GPU_RAY_TRACING_EXTENSION

VkIndexType gpuIndexTypeToVkIndexType(INDEX_TYPE indexType)
{
    switch (indexType)
    {
    case INDEX_TYPE_UINT16: return VK_INDEX_TYPE_UINT16;
    case INDEX_TYPE_UINT32: return VK_INDEX_TYPE_UINT32;
    default: return VK_INDEX_TYPE_UINT32;
    }
}

VkAccelerationStructureBuildGeometryInfoKHR gpuBuildInfoToVkBuildInfo(GpuAccelerationStructureDesc desc, std::vector<VkAccelerationStructureGeometryKHR>& outGeometries)
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    outGeometries.clear();

    if (desc.type == TYPE_BOTTOM_LEVEL)
    {
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        if (desc.blasDesc.type == GEOMETRY_TYPE_TRIANGLES)
        {
            for (const auto& triangleDesc : desc.blasDesc.triangles)
            {
                VkAccelerationStructureGeometryKHR geometry = {};
                geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

                auto& triangles = geometry.geometry.triangles;
                triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                triangles.vertexFormat = gpuFormatToVkFormat(triangleDesc.vertexFormat);
                triangles.vertexData.deviceAddress = reinterpret_cast<VkDeviceAddress>(triangleDesc.vertexDataGpu);
                triangles.vertexStride = triangleDesc.vertexStride;
                triangles.maxVertex = triangleDesc.vertexCount > 0 ? triangleDesc.vertexCount - 1 : 0;

                if (triangleDesc.indexDataGpu != nullptr)
                {
                    triangles.indexType = gpuIndexTypeToVkIndexType(triangleDesc.indexType);
                    triangles.indexData.deviceAddress = reinterpret_cast<VkDeviceAddress>(triangleDesc.indexDataGpu);
                }
                else
                {
                    triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
                    triangles.indexData.deviceAddress = 0;
                }

                if (triangleDesc.transformDataGpu != nullptr)
                {
                    triangles.transformData.deviceAddress = reinterpret_cast<VkDeviceAddress>(triangleDesc.transformDataGpu);
                }
                else
                {
                    triangles.transformData.deviceAddress = 0;
                }

                outGeometries.push_back(geometry);
            }
        }
        else // GEOMETRY_TYPE_AABBS
        {
            for (const auto& aabbDesc : desc.blasDesc.aabbs)
            {
                VkAccelerationStructureGeometryKHR geometry = {};
                geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
                geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

                auto& aabbs = geometry.geometry.aabbs;
                aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
                aabbs.data.deviceAddress = reinterpret_cast<VkDeviceAddress>(aabbDesc.aabbDataGpu);
                aabbs.stride = aabbDesc.stride;

                outGeometries.push_back(geometry);
            }
        }
    }
    else // TYPE_TOP_LEVEL
    {
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

        // For TLAS, we need to set up instance geometry
        // The instance data should be provided as a buffer of VkAccelerationStructureInstanceKHR
        VkAccelerationStructureGeometryKHR geometry = {};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        auto& instances = geometry.geometry.instances;
        instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instances.arrayOfPointers = desc.tlasDesc.arrayOfPointers ? VK_TRUE : VK_FALSE;
        instances.data.deviceAddress = reinterpret_cast<VkDeviceAddress>(desc.tlasDesc.instancesGpu);

        outGeometries.push_back(geometry);
    }

    buildInfo.geometryCount = static_cast<uint32_t>(outGeometries.size());
    buildInfo.pGeometries = outGeometries.data();

    return buildInfo;
}

GpuAccelerationStructureSizes gpuAccelerationStructureSizes(GpuAccelerationStructureDesc desc)
{
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = gpuBuildInfoToVkBuildInfo(desc, geometries);
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    vulkan->dispatchTable.getAccelerationStructureBuildSizesKHR(
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        &desc.buildRange->primitiveCount,
        &sizeInfo
    );

    return {
        sizeInfo.accelerationStructureSize,
        sizeInfo.updateScratchSize,
        sizeInfo.buildScratchSize
    };
}

GpuAccelerationStructure gpuCreateAccelerationStructure(GpuAccelerationStructureDesc desc, void *ptrGpu, uint64_t size)
{
    auto alloc = vulkan->findAllocation(reinterpret_cast<VkDeviceAddress>(ptrGpu));
    if (alloc.buffer == VK_NULL_HANDLE)
    {
        return nullptr;
    }

    VkAccelerationStructureCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = alloc.buffer;
    createInfo.offset = reinterpret_cast<VkDeviceAddress>(ptrGpu) - alloc.address;
    createInfo.size = size;
    createInfo.type = (desc.type == TYPE_BOTTOM_LEVEL) ? VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    VkAccelerationStructureKHR vkAs;
    vulkan->dispatchTable.createAccelerationStructureKHR(
        &createInfo,
        nullptr,
        &vkAs
    );

    auto as = new GpuAccelerationStructure_T();

    as->buildInfo = gpuBuildInfoToVkBuildInfo(desc, as->geometries);
    as->buildInfo.dstAccelerationStructure = vkAs;

    as->buildRange.firstVertex = desc.buildRange->firstVertex;
    as->buildRange.primitiveCount = desc.buildRange->primitiveCount;
    as->buildRange.primitiveOffset = desc.buildRange->primitiveOffset;
    as->buildRange.firstVertex = desc.buildRange->firstVertex;
    as->buildRange.transformOffset = desc.buildRange->transformOffset;

    return as;
}

void gpuBuildAccelerationStructures(GpuCommandBuffer cb, Span<GpuAccelerationStructure> as, void *scratchGpu, MODE mode)
{
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> buildRanges;
    for (const auto& a : as)
    {
        if (mode == MODE_UPDATE)
        {
            a->buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            a->buildInfo.srcAccelerationStructure = a->buildInfo.dstAccelerationStructure;
        }
        else
        {
            a->buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            a->buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
        }

        a->buildInfo.scratchData.deviceAddress = reinterpret_cast<VkDeviceAddress>(scratchGpu);

        buildInfos.push_back(a->buildInfo);
        buildRanges.push_back(&a->buildRange);
    }

    vulkan->dispatchTable.cmdBuildAccelerationStructuresKHR(
        cb->commandBuffer,
        as.size(),
        buildInfos.data(),
        buildRanges.data()
    );
}

void gpuDestroyAccelerationStructure(GpuAccelerationStructure as)
{
    vulkan->dispatchTable.destroyAccelerationStructureKHR(as->buildInfo.dstAccelerationStructure, nullptr);
    delete as;
}
#endif // GPU_RAY_TRACING_EXTENSION