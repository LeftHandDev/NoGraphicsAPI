#define GPU_EXPOSE_INTERNAL
#include "NoGraphicsAPI.h"
#include "External/VkBootstrap.h"

#include "Config.h"
#include "Shaders/PatchDescriptors.h"

#include <map>
#include <vector>
#include <fstream>
#include <filesystem>
#include "NoGraphicsAPI_Impl.h"

std::vector<uint8_t> gpuHiddenLoadIR(const std::filesystem::path &path)
{
    std::ifstream file { path, std::ios::binary | std::ios::ate };
    if (!file.is_open())
    {
        return {};
    }
    auto size = file.tellg();
    std::vector<uint8_t> buffer(size);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

struct GpuPipeline_T 
{ 
    VkPipeline pipeline; 
    VkPipelineBindPoint bindPoint;
    GpuDevice device;
};
struct GpuTexture_T 
{ 
    GpuTextureDesc desc = {}; 
    VkImage image = VK_NULL_HANDLE; 
    VkImageView view = VK_NULL_HANDLE;
    GpuDevice device;
};
struct VulkanDevice;
struct GpuDevice_T { VulkanDevice* vulkanDevice = nullptr; };
struct GpuDepthStencilState_T { GpuDepthStencilDesc desc; };
struct GpuBlendState_T { GpuBlendDesc desc; };
struct GpuQueue_T { VkQueue queue; GpuDevice device; };
struct GpuCommandBuffer_T { VkCommandBuffer commandBuffer; GpuDevice device; };
struct GpuSemaphore_T { VkSemaphore semaphore; GpuDevice device; };
#ifdef GPU_SURFACE_EXTENSION
struct GpuSurface_T { VkSurfaceKHR surface = VK_NULL_HANDLE; std::vector<FORMAT> formats; GpuDevice device; };
struct GpuSwapchain_T 
{ 
    VkSwapchainKHR swapchain = VK_NULL_HANDLE; 
    VkQueue presentQueue = VK_NULL_HANDLE;
    uint32_t imageIndex = 0; 
    GpuTextureDesc desc = {};
    std::vector<GpuTexture> images;
    std::vector<VkSemaphore> presentSemaphores;
    GpuDevice device;
};
#endif // GPU_SURFACE_EXTENSION
#ifdef GPU_RAY_TRACING_EXTENSION
struct GpuAccelerationStructure_T 
{ 
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges; 
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    GpuDevice device;
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
    case VK_FORMAT_R32G32_SFLOAT: return FORMAT_RG32_FLOAT;
    case VK_FORMAT_R32G32B32_SFLOAT: return FORMAT_RGB32_FLOAT;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return FORMAT_RGBA32_FLOAT;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return FORMAT_RGBA16_FLOAT;
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
    case FORMAT_RG32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case FORMAT_RGB32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case FORMAT_RGBA32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case FORMAT_RGBA16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
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
    if (usage & USAGE_TRANSFER_SRC)
    {
        result = static_cast<uint32_t>(VK_IMAGE_USAGE_TRANSFER_SRC_BIT) | result;
    }

    return static_cast<VkImageUsageFlagBits>(result);
}

VkBlendOp gpuBlendOpToVkBlendOp(BLEND blend)
{
    switch (blend)
    {
    case BLEND_ADD: return VK_BLEND_OP_ADD;
    case BLEND_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
    case BLEND_REV_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case BLEND_MIN: return VK_BLEND_OP_MIN;
    case BLEND_MAX: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

VkBlendFactor gpuFactorToVkFactor(FACTOR factor)
{
    switch (factor)
    {
    case FACTOR_ZERO: return VK_BLEND_FACTOR_ZERO;
    case FACTOR_ONE: return VK_BLEND_FACTOR_ONE;
    case FACTOR_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
    case FACTOR_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
    case FACTOR_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
    default: return VK_BLEND_FACTOR_ONE;
    }
}

VkCompareOp gpuOpToVkCompareOp(OP op)
{
    switch (op)
    {
    case OP_NEVER: return VK_COMPARE_OP_NEVER;
    case OP_LESS: return VK_COMPARE_OP_LESS;
    case OP_EQUAL: return VK_COMPARE_OP_EQUAL;
    case OP_LESS_EQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case OP_GREATER: return VK_COMPARE_OP_GREATER;
    case OP_NOT_EQUAL: return VK_COMPARE_OP_NOT_EQUAL;
    case OP_GREATER_EQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case OP_ALWAYS: return VK_COMPARE_OP_ALWAYS;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

VkStencilOp gpuOpToVkStencilOp(OP op)
{
    switch (op)
    {
    case OP_KEEP: return VK_STENCIL_OP_KEEP;
    case OP_ZERO: return VK_STENCIL_OP_ZERO;
    default: return VK_STENCIL_OP_KEEP;
    }
}

struct Allocation
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    void* ptr = nullptr;
};

struct VulkanInstance
{
    vkb::Instance instance;
    vkb::InstanceDispatchTable instanceDispatchTable;
    std::vector<const char*> requiredDeviceExtensions;
    std::vector<GpuDeviceDesc> deviceDescs;

    static VulkanInstance* create()
    {
        VulkanInstance* inst = new VulkanInstance();

        std::vector<const char*> requiredInstanceExtensions = {};
        inst->requiredDeviceExtensions = {
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
        inst->requiredDeviceExtensions.push_back(VK_KHR_DISPLAY_SWAPCHAIN_EXTENSION_NAME);
#endif
        inst->requiredDeviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#endif // GPU_SURFACE_EXTENSION
#ifdef GPU_RAY_TRACING_EXTENSION
        inst->requiredDeviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        inst->requiredDeviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        inst->requiredDeviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        inst->requiredDeviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
#endif // GPU_RAY_TRACING_EXTENSION

        vkb::InstanceBuilder instanceBuilder;

        auto instanceRet = instanceBuilder
            .request_validation_layers(true)
            .require_api_version(VK_API_VERSION_1_4)
            .enable_extensions(requiredInstanceExtensions)
            .build();

        if (!instanceRet.has_value())
        {
            delete inst;
            return nullptr;
        }

        inst->instance = instanceRet.value();
        inst->instanceDispatchTable = inst->instance.make_table();

        return inst;
    }

    ~VulkanInstance()
    {
        vkb::destroy_instance(instance);
    }
};

VulkanInstance* vulkanInstance = nullptr;

struct VulkanDevice
{
    // Back-reference to the global instance
    VulkanInstance* inst = nullptr;

    // VkBootstrap objects
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
    GpuPipeline patchDescriptorsPipeline = nullptr;
    void* descriptorDataCpu = nullptr; // the raw descriptor data, possibly out of order
    void* rwDescriptorDataCpu = nullptr; // the raw read/write descriptor data, possibly out of order
    void* patchedDescriptorDataCpu = nullptr; // the temporary patched descriptor data
    void* rwPatchedDescriptorDataCpu = nullptr; // the temporary patched read/write descriptor data
    PatchDescriptorsData* patchDescriptorsDataCpu = nullptr;
    uint32_t descriptorsUsed = 0;
    uint32_t rwDescriptorsUsed = 0;

    static VulkanDevice* createVulkan(uint32_t deviceIndex)
    {
        if (vulkanInstance == nullptr)
        {
            return nullptr;
        }

        VulkanDevice* vulkanDevice = new VulkanDevice();
        vulkanDevice->inst = vulkanInstance;

        vkb::PhysicalDeviceSelector deviceSelector{ vulkanInstance->instance };
        auto physicalDevicesRet = deviceSelector
            .add_required_extensions(vulkanInstance->requiredDeviceExtensions)
            .defer_surface_initialization()
            .select_devices();
        if (!physicalDevicesRet.has_value() || deviceIndex >= physicalDevicesRet.value().size())
        {
            delete vulkanDevice;
            return nullptr;
        }

        vulkanDevice->physicalDevice = physicalDevicesRet.value()[deviceIndex];

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
        physicalDeviceVulkan12Features.shaderInt8 = VK_TRUE;

        vulkanDevice->physicalDevice.features.shaderInt64 = VK_TRUE;

#ifdef GPU_RAY_TRACING_EXTENSION
#endif // GPU_RAY_TRACING_EXTENSION
        vulkanInstance->instanceDispatchTable.getPhysicalDeviceMemoryProperties(vulkanDevice->physicalDevice, &vulkanDevice->memoryProperties);

        vulkanDevice->descriptorBufferProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;
        vulkanDevice->physicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        vulkanDevice->physicalDeviceProperties2.pNext = &vulkanDevice->descriptorBufferProperties;
        vulkanInstance->instanceDispatchTable.getPhysicalDeviceProperties2(vulkanDevice->physicalDevice, &vulkanDevice->physicalDeviceProperties2);

        vkb::DeviceBuilder deviceBuilder{ vulkanDevice->physicalDevice };
        deviceBuilder
            .add_pNext(&physicalDeviceVulkan12Features)
            .add_pNext(&physicalDeviceVulkan13Features)
            .add_pNext(&descriptorBufferFeatures);
#ifdef GPU_RAY_TRACING_EXTENSION
        deviceBuilder
            .add_pNext(&rayQueryFeatures)
            .add_pNext(&accelerationStructureFeatures);
#endif // GPU_RAY_TRACING_EXTENSION
        auto deviceRet = deviceBuilder.build();
        vulkanDevice->device = deviceRet.value();
        vulkanDevice->dispatchTable = vulkanDevice->device.make_table();

        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = vulkanDevice->device.get_queue_index(vkb::QueueType::graphics).value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vulkanDevice->dispatchTable.createCommandPool(&poolInfo, nullptr, &vulkanDevice->commandPool);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        vulkanDevice->dispatchTable.createSampler(&samplerInfo, nullptr, &vulkanDevice->defaultSampler);

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vulkanDevice->dispatchTable.createFence(&fenceInfo, nullptr, &vulkanDevice->acquireFence);

        vulkanDevice->createPipelineLayout();

        return vulkanDevice;
    }

    ~VulkanDevice()
    {
        dispatchTable.deviceWaitIdle();

        for (auto& [key, commandBuffers] : submittedCommandBuffers)
        {
            dispatchTable.freeCommandBuffers(commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());
        }
        submittedCommandBuffers.clear();

        if (patchDescriptorsPipeline != nullptr)
        {
            dispatchTable.destroyPipeline(patchDescriptorsPipeline->pipeline, nullptr);
            delete patchDescriptorsPipeline;
        }

        if (descriptorDataCpu != nullptr)
        { 
            freeAllocation(findAllocation(descriptorDataCpu));
        }

        if (rwDescriptorDataCpu != nullptr)
        {
            freeAllocation(findAllocation(rwDescriptorDataCpu));
        }

        if (patchedDescriptorDataCpu != nullptr)
        { 
            freeAllocation(findAllocation(patchedDescriptorDataCpu));
        }

        if (rwPatchedDescriptorDataCpu != nullptr) 
        {
            freeAllocation(findAllocation(rwPatchedDescriptorDataCpu));
        }
        
        if (patchDescriptorsDataCpu != nullptr) 
        {
            freeAllocation(findAllocation(patchDescriptorsDataCpu));
        }

        if (samplerDescriptors.buffer != VK_NULL_HANDLE)
        {
            freeAllocation(samplerDescriptors);
        }

        dispatchTable.destroyCommandPool(commandPool, nullptr);
        dispatchTable.destroySampler(defaultSampler, nullptr);
        dispatchTable.destroyFence(acquireFence, nullptr);
        dispatchTable.destroyDescriptorSetLayout(textureSetLayout, nullptr);
        dispatchTable.destroyDescriptorSetLayout(rwTextureSetLayout, nullptr);
        dispatchTable.destroyDescriptorSetLayout(samplerSetLayout, nullptr);
        dispatchTable.destroyPipelineLayout(layout[VK_PIPELINE_BIND_POINT_GRAPHICS], nullptr);
        dispatchTable.destroyPipelineLayout(layout[VK_PIPELINE_BIND_POINT_COMPUTE], nullptr);
        // dispatchTable.destroyPipelineLayout(layout[VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR], nullptr);
        vkb::destroy_device(device);
    }

    bool descriptorsNeedPatching()
    {
        return descriptorBufferProperties.sampledImageDescriptorSize != sizeof(GpuTextureDescriptor) ||
               descriptorBufferProperties.storageImageDescriptorSize != sizeof(GpuTextureDescriptor);
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

void* gpuVulkanInstance()
{
    return vulkanInstance->instance.instance;
}

GpuSurface gpuCreateSurface(void *vulkanSurface)
{
    return new GpuSurface_T { static_cast<VkSurfaceKHR>(vulkanSurface), {} };
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

RESULT gpuCreateInstance()
{
    if (vulkanInstance == nullptr)
    {
        vulkanInstance = VulkanInstance::create();
        if (vulkanInstance == nullptr)
        {
            return RESULT_FAILURE;
        }
    }

    return RESULT_SUCCESS;
}

void gpuDestroyInstance()
{
    if (vulkanInstance != nullptr)
    {
        delete vulkanInstance;
        vulkanInstance = nullptr;
    }
}

static void gpuPopulateDeviceDescs()
{
    if (!vulkanInstance->deviceDescs.empty())
    {
        return;
    }

    vkb::PhysicalDeviceSelector deviceSelector{ vulkanInstance->instance };
    auto physicalDevicesRet = deviceSelector
        .add_required_extensions(vulkanInstance->requiredDeviceExtensions)
        .defer_surface_initialization()
        .select_devices();

    if (!physicalDevicesRet.has_value())
    {
        return;
    }

    auto& devices = physicalDevicesRet.value();
    for (uint32_t i = 0; i < static_cast<uint32_t>(devices.size()); i++)
    {
        VkPhysicalDeviceProperties props;
        vulkanInstance->instanceDispatchTable.getPhysicalDeviceProperties(devices[i], &props);

        VkPhysicalDeviceMemoryProperties memProps;
        vulkanInstance->instanceDispatchTable.getPhysicalDeviceMemoryProperties(devices[i], &memProps);

        uint64_t dedicatedMemory = 0;
        for (uint32_t j = 0; j < memProps.memoryHeapCount; j++)
        {
            if (memProps.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
                dedicatedMemory += memProps.memoryHeaps[j].size;
            }
        }

        GpuDeviceDesc desc = {};
        memset(desc.name, 0, sizeof(desc.name));
        strncpy(desc.name, props.deviceName, sizeof(desc.name) - 1);
        desc.vendorID = props.vendorID;
        desc.dedicatedMemory = dedicatedMemory;
        desc.discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
        vulkanInstance->deviceDescs.push_back(desc);
    }
}

uint32_t gpuDeviceCount()
{
    if (vulkanInstance == nullptr)
    {
        return 0;
    }

    gpuPopulateDeviceDescs();
    return static_cast<uint32_t>(vulkanInstance->deviceDescs.size());
}

GpuDeviceDesc gpuDeviceDesc(uint32_t index)
{
    if (vulkanInstance == nullptr)
    {
        return {};
    }

    gpuPopulateDeviceDescs();

    if (index >= static_cast<uint32_t>(vulkanInstance->deviceDescs.size()))
    {
        return {};
    }

    return vulkanInstance->deviceDescs[index];
}

GpuDevice gpuCreateDevice(uint32_t deviceIndex)
{
    if (vulkanInstance == nullptr)
    {
        return nullptr;
    }

    VulkanDevice* vulkanDevice = VulkanDevice::createVulkan(deviceIndex);
    if (vulkanDevice == nullptr)
    {
        return nullptr;
    }

    GpuDevice device = new GpuDevice_T{ vulkanDevice };
    return device;
}

void gpuDestroyDevice(GpuDevice device)
{
    if (device != nullptr)
    {
        delete device->vulkanDevice;
        delete device;
    }
}

void* gpuMalloc(GpuDevice device, size_t bytes, MEMORY memory)
{
    return gpuMalloc(device, bytes, GPU_DEFAULT_ALIGNMENT, memory);
}

void* gpuMallocHidden(VulkanDevice* vulkanDevice, size_t bytes, size_t align, MEMORY memory, bool sampler = false)
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

        auto alloc = vulkanDevice->createAllocation(bytes, usage, properties, align);
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
        auto alloc = vulkanDevice->createAllocation(bytes, usage, properties, align);
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

        auto alloc = vulkanDevice->createAllocation(bytes, usage, properties, align);
        return alloc.ptr;    
    }
    };

    return nullptr;
}

void* gpuMalloc(GpuDevice device, size_t bytes, size_t align, MEMORY memory)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    return gpuMallocHidden(vulkanDevice, bytes, align, memory);
}

void gpuFree(GpuDevice device, void *ptr)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    for (auto& alloc : vulkanDevice->allocations)
    {
        if (alloc.ptr == ptr || reinterpret_cast<VkDeviceAddress>(ptr) == alloc.address)
        {
            vulkanDevice->freeAllocation(alloc);
            return;
        }
    }
}

void* gpuHostToDevicePointer(GpuDevice device, void *ptr)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    Allocation alloc = vulkanDevice->findAllocation(ptr);
    if (alloc.buffer != VK_NULL_HANDLE)
    {
        return reinterpret_cast<void*>(alloc.address + (static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(alloc.ptr)));
    }
    return nullptr;
}

GpuTextureSizeAlign gpuTextureSizeAlign(GpuDevice device, GpuTextureDesc desc)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    VkImage image = vulkanDevice->createImage(desc);

    VkMemoryRequirements imageMemoryRequirements = {};
    vulkanDevice->dispatchTable.getImageMemoryRequirements(image, &imageMemoryRequirements);
    vulkanDevice->dispatchTable.destroyImage(image, nullptr);

    return { imageMemoryRequirements.size, imageMemoryRequirements.alignment };
}

GpuTexture gpuCreateTexture(GpuDevice device, GpuTextureDesc desc, void* ptrGpu)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    Allocation alloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(ptrGpu));

    VkImage image = vulkanDevice->createImage(desc);

    VkDeviceSize offset = reinterpret_cast<VkDeviceAddress>(ptrGpu) - alloc.address;
    vulkanDevice->dispatchTable.bindImageMemory(image, alloc.memory, offset);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = desc.type == TEXTURE_1D ? VK_IMAGE_VIEW_TYPE_1D :
                        desc.type == TEXTURE_2D ? VK_IMAGE_VIEW_TYPE_2D :
                        desc.type == TEXTURE_3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_MAX_ENUM;
    viewInfo.format = gpuFormatToVkFormat(desc.format);
    viewInfo.subresourceRange.aspectMask = (desc.usage & USAGE_DEPTH_STENCIL_ATTACHMENT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = desc.mipCount;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = desc.layerCount;

    VkImageView imageView = VK_NULL_HANDLE;
    vulkanDevice->dispatchTable.createImageView(&viewInfo, nullptr, &imageView);

    return new GpuTexture_T { desc, image, imageView, device};
}

void gpuDestroyTexture(GpuTexture texture)
{
    if (texture == nullptr)
    {
        return;
    }

    VulkanDevice* vulkanDevice = texture->device->vulkanDevice;
    vulkanDevice->dispatchTable.destroyImageView(texture->view, nullptr);
    vulkanDevice->dispatchTable.destroyImage(texture->image, nullptr);
    delete texture;
}

GpuTextureDescriptor gpuTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc)
{
    VulkanDevice* vulkanDevice = texture->device->vulkanDevice;
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = vulkanDevice->defaultSampler;
    imageInfo.imageView = texture->view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorGetInfoEXT descriptorGetInfo = {};
    descriptorGetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    descriptorGetInfo.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptorGetInfo.data.pSampledImage = &imageInfo;

    std::vector<uint8_t> buffer(vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize);
    vulkanDevice->dispatchTable.getDescriptorEXT(&descriptorGetInfo, vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize, buffer.data());

    GpuTextureDescriptor descriptor = {};
    
    if (vulkanDevice->descriptorsNeedPatching())
    {
        if (!vulkanDevice->descriptorDataCpu)
        {
            vulkanDevice->descriptorDataCpu = gpuMallocHidden(vulkanDevice, vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize * vulkanDevice->descriptorCount, GPU_DEFAULT_ALIGNMENT, MEMORY_DEFAULT);
        }

        uint8_t* dest = static_cast<uint8_t*>(vulkanDevice->descriptorDataCpu) + vulkanDevice->descriptorsUsed * vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize;
        memcpy(dest, buffer.data(), vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize);
        descriptor.data[0] = vulkanDevice->descriptorsUsed * vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize; // Store byte offset in our internal buffer
        descriptor.data[1] = 0; // type 0 for read, 1 for read/write
        vulkanDevice->descriptorsUsed++;
    }
    else
    {
        memcpy(descriptor.data, buffer.data(), vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize);
    }

    return descriptor;
}

GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc)
{
    VulkanDevice* vulkanDevice = texture->device->vulkanDevice;
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = VK_NULL_HANDLE;
    imageInfo.imageView = texture->view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorGetInfoEXT descriptorGetInfo = {};
    descriptorGetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
    descriptorGetInfo.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorGetInfo.data.pStorageImage = &imageInfo;

    std::vector<uint8_t> buffer(vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize);
    vulkanDevice->dispatchTable.getDescriptorEXT(&descriptorGetInfo, vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize, buffer.data());

    GpuTextureDescriptor descriptor = {};

    if (vulkanDevice->descriptorsNeedPatching())
    {
        if (!vulkanDevice->rwDescriptorDataCpu)
        {
            vulkanDevice->rwDescriptorDataCpu = gpuMallocHidden(vulkanDevice, vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize * vulkanDevice->descriptorCount, GPU_DEFAULT_ALIGNMENT, MEMORY_DEFAULT);
        }

        uint8_t* dest = static_cast<uint8_t*>(vulkanDevice->rwDescriptorDataCpu) + vulkanDevice->rwDescriptorsUsed * vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize;
        memcpy(dest, buffer.data(), vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize);
        descriptor.data[0] = vulkanDevice->rwDescriptorsUsed * vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize; // Store byte offset in our internal buffer
        descriptor.data[1] = 1; // type 0 for read, 1 for read/write
        vulkanDevice->rwDescriptorsUsed++;
    }
    else
    {
        memcpy(descriptor.data, buffer.data(), vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize);
    }

    return descriptor;
}

GpuPipeline gpuCreateComputePipeline(GpuDevice device, ByteSpan computeIR)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = computeIR.size();
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(computeIR.data());
    VkShaderModule shaderModule;
    vulkanDevice->dispatchTable.createShaderModule(&shaderModuleCreateInfo, nullptr, &shaderModule);

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.layout = vulkanDevice->layout[VK_PIPELINE_BIND_POINT_COMPUTE];
    pipelineCreateInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pName = "main";

    VkPipeline pipeline;
    vulkanDevice->dispatchTable.createComputePipelines(VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);
    vulkanDevice->dispatchTable.destroyShaderModule(shaderModule, nullptr);

    return new GpuPipeline_T { pipeline, VK_PIPELINE_BIND_POINT_COMPUTE, device };
}

VkPipeline gpuCreateGraphicsPipelineInternal(VulkanDevice* vulkanDevice, ByteSpan vertexIR, ByteSpan meshletIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    bool vertex = vertexIR.size() > 0;
    ByteSpan actualIR = vertex ? vertexIR : meshletIR;

    VkShaderModuleCreateInfo vertexShaderModuleCreateInfo = {};
    vertexShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vertexShaderModuleCreateInfo.codeSize = actualIR.size();
    vertexShaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(actualIR.data());
    VkShaderModule vertexShaderModule;
    vulkanDevice->dispatchTable.createShaderModule(&vertexShaderModuleCreateInfo, nullptr, &vertexShaderModule);

    VkShaderModuleCreateInfo pixelShaderModuleCreateInfo = {};
    pixelShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    pixelShaderModuleCreateInfo.codeSize = pixelIR.size();
    pixelShaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(pixelIR.data());
    VkShaderModule pixelShaderModule;
    vulkanDevice->dispatchTable.createShaderModule(&pixelShaderModuleCreateInfo, nullptr, &pixelShaderModule);

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
            blendAttachment.srcColorBlendFactor = gpuFactorToVkFactor(blendDesc.srcColorFactor);
            blendAttachment.dstColorBlendFactor = gpuFactorToVkFactor(blendDesc.dstColorFactor);
            blendAttachment.colorBlendOp = gpuBlendOpToVkBlendOp(blendDesc.colorOp);
            blendAttachment.srcAlphaBlendFactor = gpuFactorToVkFactor(blendDesc.srcAlphaFactor);
            blendAttachment.dstAlphaBlendFactor = gpuFactorToVkFactor(blendDesc.dstAlphaFactor);
            blendAttachment.alphaBlendOp = gpuBlendOpToVkBlendOp(blendDesc.alphaOp);
            blendAttachment.colorWriteMask = 
                ((blendDesc.colorWriteMask & 0x1) ? VK_COLOR_COMPONENT_R_BIT : 0) |
                ((blendDesc.colorWriteMask & 0x2) ? VK_COLOR_COMPONENT_G_BIT : 0) |
                ((blendDesc.colorWriteMask & 0x4) ? VK_COLOR_COMPONENT_B_BIT : 0) |
                ((blendDesc.colorWriteMask & 0x8) ? VK_COLOR_COMPONENT_A_BIT : 0);
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
                ((target.writeMask & 0x1) ? VK_COLOR_COMPONENT_R_BIT : 0) |
                ((target.writeMask & 0x2) ? VK_COLOR_COMPONENT_G_BIT : 0) |
                ((target.writeMask & 0x4) ? VK_COLOR_COMPONENT_B_BIT : 0) |
                ((target.writeMask & 0x8) ? VK_COLOR_COMPONENT_A_BIT : 0);
        }

        blendAttachments.push_back(blendAttachment);
    }

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = desc.colorTargets.size();
    pipelineRenderingInfo.pColorAttachmentFormats = colorFormats.data();
    pipelineRenderingInfo.depthAttachmentFormat = gpuFormatToVkFormat(desc.depthFormat);
    pipelineRenderingInfo.stencilAttachmentFormat = gpuFormatToVkFormat(desc.stencilFormat);

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

    VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
    depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilState.depthTestEnable = (desc.depthFormat != FORMAT_NONE) ? VK_TRUE : VK_FALSE;
    depthStencilState.depthWriteEnable = (desc.depthFormat != FORMAT_NONE) ? VK_TRUE : VK_FALSE;
    depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencilState.depthBoundsTestEnable = VK_FALSE;
    depthStencilState.stencilTestEnable = (desc.stencilFormat != FORMAT_NONE) ? VK_TRUE : VK_FALSE;
    depthStencilState.minDepthBounds = 0.0f;
    depthStencilState.maxDepthBounds = 1.0f;

    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_BIAS
    };
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = &pipelineRenderingInfo;
    pipelineCreateInfo.layout = vulkanDevice->layout[VK_PIPELINE_BIND_POINT_GRAPHICS];
    pipelineCreateInfo.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyInfo;
    pipelineCreateInfo.pColorBlendState = &blendState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = 2;

    VkPipeline pipeline;
    vulkanDevice->dispatchTable.createGraphicsPipelines(VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);
    vulkanDevice->dispatchTable.destroyShaderModule(vertexShaderModule, nullptr);
    vulkanDevice->dispatchTable.destroyShaderModule(pixelShaderModule, nullptr);

    return pipeline;
}

GpuPipeline gpuCreateGraphicsPipeline(GpuDevice device, ByteSpan vertexIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    VkPipeline pipeline = gpuCreateGraphicsPipelineInternal(vulkanDevice, vertexIR, ByteSpan{}, pixelIR, desc);
    return new GpuPipeline_T { pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS, device };
}

GpuPipeline gpuCreateGraphicsMeshletPipeline(GpuDevice device, ByteSpan meshletIR, ByteSpan pixelIR, GpuRasterDesc desc)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    VkPipeline pipeline = gpuCreateGraphicsPipelineInternal(vulkanDevice, ByteSpan{}, meshletIR, pixelIR, desc);
    return new GpuPipeline_T { pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS, device};
}

void gpuFreePipeline(GpuPipeline pipeline)
{
    VulkanDevice* vulkanDevice = pipeline->device->vulkanDevice;
    vulkanDevice->dispatchTable.destroyPipeline(pipeline->pipeline, nullptr);
    delete pipeline;
}

GpuDepthStencilState gpuCreateDepthStencilState(GpuDepthStencilDesc desc)
{
    return new GpuDepthStencilState_T { desc };
}

GpuBlendState gpuCreateBlendState(GpuBlendDesc desc)
{
    return new GpuBlendState_T { desc };
}

void gpuFreeDepthStencilState(GpuDepthStencilState state)
{
    if (state != nullptr)
    {
        delete state;
    }
}

void gpuFreeBlendState(GpuBlendState state)
{
    if (state != nullptr)
    {
        delete state;
    }
}

GpuQueue gpuCreateQueue(GpuDevice device)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    GpuQueue queue = new GpuQueue_T { vulkanDevice->device.get_queue(vkb::QueueType::graphics).value(), device};
    return queue;
}

void gpuDestroyQueue(GpuQueue queue)
{
    delete queue;
}

GpuCommandBuffer gpuStartCommandRecording(GpuQueue queue)
{
    VulkanDevice* vulkanDevice = queue->device->vulkanDevice;
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vulkanDevice->commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vulkanDevice->dispatchTable.allocateCommandBuffers(&allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vulkanDevice->dispatchTable.beginCommandBuffer(commandBuffer, &beginInfo);

    return new GpuCommandBuffer_T { commandBuffer, queue->device };
}

void gpuSubmit(GpuQueue queue, Span<GpuCommandBuffer> commandBuffers, GpuSemaphore semaphore, uint64_t value)
{
    VulkanDevice* vulkanDevice = queue->device->vulkanDevice;
    std::vector<VkCommandBuffer> vkCommandBuffers;
    for (auto cb : commandBuffers)
    {
        vkCommandBuffers.push_back(cb->commandBuffer);
        vulkanDevice->dispatchTable.endCommandBuffer(cb->commandBuffer);
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

    vulkanDevice->dispatchTable.queueSubmit(queue->queue, 1, &submitInfo, VK_NULL_HANDLE);

    vulkanDevice->submittedCommandBuffers[{ semaphore->semaphore, value }] = vkCommandBuffers;

    for (auto cb : commandBuffers)
    {
        vulkanDevice->currentPipeline.erase(cb);
        delete cb; // Frees the wrapper, but not the actual VkCommandBuffer
    }
}

GpuSemaphore gpuCreateSemaphore(GpuDevice device, uint64_t initValue)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    VkSemaphoreTypeCreateInfo semaphoreTypeInfo = {};
    semaphoreTypeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    semaphoreTypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    semaphoreTypeInfo.initialValue = initValue;

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &semaphoreTypeInfo;

    VkSemaphore semaphore;
    vulkanDevice->dispatchTable.createSemaphore(&semaphoreInfo, nullptr, &semaphore);

    return new GpuSemaphore_T { semaphore, device };
}

void gpuWaitSemaphore(GpuSemaphore sema, uint64_t value, uint64_t timeout)
{
    VulkanDevice* vulkanDevice = sema->device->vulkanDevice;
    VkSemaphoreWaitInfo waitInfo = {};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &sema->semaphore;
    waitInfo.pValues = &value;

    vulkanDevice->dispatchTable.waitSemaphores(&waitInfo, timeout);

    // Free command buffers for this semaphore, and any earlier values if they exist
    for (size_t i = value; i > 0; i--)
    {
        if (vulkanDevice->submittedCommandBuffers.find({sema->semaphore, i}) != vulkanDevice->submittedCommandBuffers.end())
        {
            auto& cmdBuffers = vulkanDevice->submittedCommandBuffers[{sema->semaphore, i}];
            vulkanDevice->dispatchTable.freeCommandBuffers(vulkanDevice->commandPool, static_cast<uint32_t>(cmdBuffers.size()), cmdBuffers.data());
            vulkanDevice->submittedCommandBuffers.erase({sema->semaphore, i});
        }
        else
        {
            break;
        }
    }
}

void gpuDestroySemaphore(GpuSemaphore sema)
{
    VulkanDevice* vulkanDevice = sema->device->vulkanDevice;
    vulkanDevice->dispatchTable.destroySemaphore(sema->semaphore, nullptr);
    delete sema;
}

void gpuMemCpy(GpuCommandBuffer cb, void* destGpu, void* srcGpu, uint64_t size)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    Allocation src = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(srcGpu));
    Allocation dst = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(destGpu));

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = reinterpret_cast<VkDeviceAddress>(srcGpu) - src.address;
    copyRegion.dstOffset = reinterpret_cast<VkDeviceAddress>(destGpu) - dst.address;
    copyRegion.size = size;

    vulkanDevice->dispatchTable.cmdCopyBuffer(cb->commandBuffer, src.buffer, dst.buffer, 1, &copyRegion);
}

void gpuCopyToTexture(GpuCommandBuffer cb, void* srcGpu, GpuTexture texture)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    Allocation src = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(srcGpu));

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

    vulkanDevice->dispatchTable.cmdCopyBufferToImage(cb->commandBuffer, src.buffer, texture->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void gpuCopyFromTexture(GpuCommandBuffer cb, void* destGpu, GpuTexture texture)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    Allocation dst = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(destGpu));

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

    vulkanDevice->dispatchTable.cmdCopyImageToBuffer(cb->commandBuffer, texture->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buffer, 1, &region);
}

void gpuBlitTexture(GpuCommandBuffer cb, GpuTexture destTexture, GpuTexture srcTexture)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
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

    vulkanDevice->dispatchTable.cmdBlitImage(
        cb->commandBuffer,
        srcTexture->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        destTexture->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &blit,
        VK_FILTER_NEAREST
    );
}

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer cb, void *ptrGpu)
{
    GpuDevice device = cb->device;
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    auto alloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(ptrGpu));

    // TODO: support samplers, for now we just bind a default sampler
    if (vulkanDevice->samplerDescriptors.buffer == VK_NULL_HANDLE)
    {
        auto cpu = gpuMallocHidden(vulkanDevice,
            vulkanDevice->samplerDescriptorSetLayoutSize, 
            vulkanDevice->descriptorBufferProperties.descriptorBufferOffsetAlignment, 
            MEMORY_DEFAULT, 
            true // sampler
        );

        vulkanDevice->samplerDescriptors = vulkanDevice->findAllocation(cpu);

        VkDescriptorGetInfoEXT samplerDescriptorGetInfo = {};
        samplerDescriptorGetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT;
        samplerDescriptorGetInfo.type = VK_DESCRIPTOR_TYPE_SAMPLER;
        samplerDescriptorGetInfo.data.pSampler = &vulkanDevice->defaultSampler;
        vulkanDevice->dispatchTable.getDescriptorEXT(&samplerDescriptorGetInfo, vulkanDevice->descriptorBufferProperties.samplerDescriptorSize, vulkanDevice->samplerDescriptors.ptr);
    }

    VkDeviceAddress address = reinterpret_cast<VkDeviceAddress>(ptrGpu);
    VkDeviceAddress rwAddress = address;

    if (vulkanDevice->descriptorsNeedPatching())
    {
        GpuPipeline currentPipeline = vulkanDevice->currentPipeline[cb];
        if (vulkanDevice->patchDescriptorsPipeline == nullptr)
        {
            auto patchDescriptorsSpv = gpuHiddenLoadIR("../Shaders/PatchDescriptors.spv");
            vulkanDevice->patchDescriptorsPipeline = gpuCreateComputePipeline(device, patchDescriptorsSpv);
        }

        if (vulkanDevice->patchedDescriptorDataCpu == nullptr)
        {
            vulkanDevice->patchedDescriptorDataCpu = 
                gpuMallocHidden(vulkanDevice, vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize * vulkanDevice->descriptorCount, 
                    vulkanDevice->descriptorBufferProperties.descriptorBufferOffsetAlignment, MEMORY_DEFAULT);
        }

        if (vulkanDevice->rwPatchedDescriptorDataCpu == nullptr)
        {
            vulkanDevice->rwPatchedDescriptorDataCpu = 
                gpuMallocHidden(vulkanDevice, vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize * vulkanDevice->descriptorCount, 
                    vulkanDevice->descriptorBufferProperties.descriptorBufferOffsetAlignment, MEMORY_DEFAULT);
        }

        auto patchedDescriptorDataGpu = gpuHostToDevicePointer(device, vulkanDevice->patchedDescriptorDataCpu);
        auto rwPatchedDescriptorDataGpu = gpuHostToDevicePointer(device, vulkanDevice->rwPatchedDescriptorDataCpu);

        gpuSetPipeline(cb, vulkanDevice->patchDescriptorsPipeline);

        if (vulkanDevice->patchDescriptorsDataCpu == nullptr)
        {
            vulkanDevice->patchDescriptorsDataCpu = static_cast<PatchDescriptorsData*>(gpuMallocHidden(vulkanDevice, sizeof(PatchDescriptorsData), GPU_DEFAULT_ALIGNMENT, MEMORY_DEFAULT));
        }

        vulkanDevice->patchDescriptorsDataCpu->numDescriptors = (alloc.size - (alloc.address - address)) / sizeof(GpuTextureDescriptor);
        vulkanDevice->patchDescriptorsDataCpu->descriptorSize = vulkanDevice->descriptorBufferProperties.sampledImageDescriptorSize;
        vulkanDevice->patchDescriptorsDataCpu->rwDescriptorSize = vulkanDevice->descriptorBufferProperties.storageImageDescriptorSize;
        vulkanDevice->patchDescriptorsDataCpu->descriptors = static_cast<Descriptor*>(ptrGpu);
        vulkanDevice->patchDescriptorsDataCpu->srcDescriptors = static_cast<uint8_t*>(gpuHostToDevicePointer(device, vulkanDevice->descriptorDataCpu));
        vulkanDevice->patchDescriptorsDataCpu->rwSrcDescriptors = static_cast<uint8_t*>(gpuHostToDevicePointer(device, vulkanDevice->rwDescriptorDataCpu));
        vulkanDevice->patchDescriptorsDataCpu->dstDescriptors = static_cast<uint8_t*>(patchedDescriptorDataGpu);
        vulkanDevice->patchDescriptorsDataCpu->rwDstDescriptors = static_cast<uint8_t*>(rwPatchedDescriptorDataGpu);

        assert(vulkanDevice->descriptorCount >= 16 && vulkanDevice->descriptorCount % 16 == 0);
        gpuDispatch(cb, gpuHostToDevicePointer(device, vulkanDevice->patchDescriptorsDataCpu), { vulkanDevice->descriptorCount / 16, 1, 1});

        gpuBarrier(cb, STAGE_COMPUTE, STAGE_COMPUTE, HAZARD_DESCRIPTORS);

        // Use patched descriptors instead of the ptrGpu
        address = reinterpret_cast<VkDeviceAddress>(patchedDescriptorDataGpu);
        rwAddress = reinterpret_cast<VkDeviceAddress>(rwPatchedDescriptorDataGpu);

        gpuSetPipeline(cb, currentPipeline);
    }

    VkDescriptorBufferBindingInfoEXT bufferBindingInfo[3] = {};
    bufferBindingInfo[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    bufferBindingInfo[0].address = address;
    bufferBindingInfo[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;

    bufferBindingInfo[1].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    bufferBindingInfo[1].address = rwAddress;
    bufferBindingInfo[1].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT;

    bufferBindingInfo[2].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT;
    bufferBindingInfo[2].address = vulkanDevice->samplerDescriptors.address;
    bufferBindingInfo[2].usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;

    vulkanDevice->dispatchTable.cmdBindDescriptorBuffersEXT(
        cb->commandBuffer,
        3,
        bufferBindingInfo
    );

    uint32_t indices[3] = { 0, 1, 2 }; // read, read/write, sampler
    VkDeviceSize offsets[3] = { 0, 0, 0 };

    vulkanDevice->dispatchTable.cmdSetDescriptorBufferOffsetsEXT(
        cb->commandBuffer,
        vulkanDevice->currentPipeline[cb]->bindPoint,
        vulkanDevice->layout[vulkanDevice->currentPipeline[cb]->bindPoint],
        0,
        3,
        indices,
        offsets
    );
}

void gpuBarrier(GpuCommandBuffer cb, STAGE before, STAGE after, HAZARD_FLAGS hazards)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
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

    if (hazards & HAZARD_DEPTH_STENCIL)
    {
        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
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

    vulkanDevice->dispatchTable.cmdPipelineBarrier(
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
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    vulkanDevice->dispatchTable.cmdBindPipeline(
        cb->commandBuffer, 
        pipeline->bindPoint,
        pipeline->pipeline
    );

    vulkanDevice->currentPipeline[cb] = pipeline;
}

void gpuSetDepthStencilState(GpuCommandBuffer cb, GpuDepthStencilState state)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    const auto& desc = state->desc;
    VkCommandBuffer cmd = cb->commandBuffer;

    // Depth state
    vulkanDevice->dispatchTable.cmdSetDepthTestEnable(cmd, (desc.depthMode & DEPTH_READ) ? VK_TRUE : VK_FALSE);
    vulkanDevice->dispatchTable.cmdSetDepthWriteEnable(cmd, (desc.depthMode & DEPTH_WRITE) ? VK_TRUE : VK_FALSE);
    vulkanDevice->dispatchTable.cmdSetDepthCompareOp(cmd, gpuOpToVkCompareOp(desc.depthTest));
    vulkanDevice->dispatchTable.cmdSetDepthBoundsTestEnable(cmd, VK_FALSE);

    // Depth bias
    bool depthBiasEnabled = desc.depthBias != 0.0f || desc.depthBiasSlopeFactor != 0.0f;
    vulkanDevice->dispatchTable.cmdSetDepthBiasEnable(cmd, depthBiasEnabled ? VK_TRUE : VK_FALSE);
    if (depthBiasEnabled)
    {
        vulkanDevice->dispatchTable.cmdSetDepthBias(cmd, desc.depthBias, desc.depthBiasClamp, desc.depthBiasSlopeFactor);
    }

    // Stencil state
    bool stencilEnabled = desc.stencilFront.test != OP_ALWAYS || desc.stencilBack.test != OP_ALWAYS;
    vulkanDevice->dispatchTable.cmdSetStencilTestEnable(cmd, stencilEnabled ? VK_TRUE : VK_FALSE);

    if (stencilEnabled)
    {
        vulkanDevice->dispatchTable.cmdSetStencilOp(
            cmd, VK_STENCIL_FACE_FRONT_BIT,
            gpuOpToVkStencilOp(desc.stencilFront.failOp),
            gpuOpToVkStencilOp(desc.stencilFront.passOp),
            gpuOpToVkStencilOp(desc.stencilFront.depthFailOp),
            gpuOpToVkCompareOp(desc.stencilFront.test)
        );
        vulkanDevice->dispatchTable.cmdSetStencilOp(
            cmd, VK_STENCIL_FACE_BACK_BIT,
            gpuOpToVkStencilOp(desc.stencilBack.failOp),
            gpuOpToVkStencilOp(desc.stencilBack.passOp),
            gpuOpToVkStencilOp(desc.stencilBack.depthFailOp),
            gpuOpToVkCompareOp(desc.stencilBack.test)
        );
        vulkanDevice->dispatchTable.cmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, desc.stencilReadMask);
        vulkanDevice->dispatchTable.cmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, desc.stencilWriteMask);
        vulkanDevice->dispatchTable.cmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_BIT, desc.stencilFront.reference);
        vulkanDevice->dispatchTable.cmdSetStencilReference(cmd, VK_STENCIL_FACE_BACK_BIT, desc.stencilBack.reference);
    }
}

void gpuSetBlendState(GpuCommandBuffer cb, GpuBlendState state)
{
    // TODO: implement
}

void gpuDispatch(GpuCommandBuffer cb, void* dataGpu, uint3 gridDimensions)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    VkDeviceAddress address = reinterpret_cast<VkDeviceAddress>(dataGpu);
    vulkanDevice->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
        vulkanDevice->layout[vulkanDevice->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(VkDeviceAddress),
        &address
    );

    vulkanDevice->dispatchTable.cmdDispatch(
        cb->commandBuffer, 
        gridDimensions.x, 
        gridDimensions.y, 
        gridDimensions.z
    );
}

void gpuDispatchIndirect(GpuCommandBuffer cb, void* dataGpu, void* gridDimensionsGpu)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    VkDeviceAddress address = reinterpret_cast<VkDeviceAddress>(dataGpu);
    vulkanDevice->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
        vulkanDevice->layout[vulkanDevice->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(VkDeviceAddress),
        &address
    );

    Allocation grid = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(gridDimensionsGpu));

    vulkanDevice->dispatchTable.cmdDispatchIndirect(
        cb->commandBuffer, 
        grid.buffer, 
        reinterpret_cast<VkDeviceAddress>(gridDimensionsGpu) - grid.address
    );
}

void gpuBeginRenderPass(GpuCommandBuffer cb, GpuRenderPassDesc desc)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
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
        colorAttachment.loadOp = desc.loadOp == LOAD_OP_LOAD ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = { 0.0f, 0.0f, 0.0f, 1.0f };
        colorAttachments.push_back(colorAttachment);
    }

    VkRenderingAttachmentInfo depthAttachment = {};
    if (desc.depthStencilTarget != nullptr)
    {
        depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depthAttachment.imageView = desc.depthStencilTarget->view;
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil = { 1.0f, 0 };
    }

    auto& colorTarget = desc.colorTargets[0];
    renderingInfo.renderArea.extent = { colorTarget->desc.dimensions.x, colorTarget->desc.dimensions.y };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = desc.colorTargets.size();
    renderingInfo.pColorAttachments = colorAttachments.data();
    renderingInfo.pDepthAttachment = desc.depthStencilTarget != nullptr ? &depthAttachment : nullptr;
    renderingInfo.pStencilAttachment = nullptr; // TODO: separate stencil support if needed

    vulkanDevice->dispatchTable.cmdBeginRendering(cb->commandBuffer, &renderingInfo);

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(colorTarget->desc.dimensions.x);
    viewport.height = static_cast<float>(colorTarget->desc.dimensions.y);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vulkanDevice->dispatchTable.cmdSetViewport(cb->commandBuffer, 0, 1, &viewport);

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = { colorTarget->desc.dimensions.x, colorTarget->desc.dimensions.y };
    vulkanDevice->dispatchTable.cmdSetScissor(cb->commandBuffer, 0, 1, &scissor);
}

void gpuEndRenderPass(GpuCommandBuffer cb)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    vulkanDevice->dispatchTable.cmdEndRendering(cb->commandBuffer);
}

void gpuDrawIndexedInstanced(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, uint32_t indexCount, uint32_t instanceCount)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    VkDeviceAddress pushConstants[3] = {
        reinterpret_cast<VkDeviceAddress>(vertexDataGpu),
        reinterpret_cast<VkDeviceAddress>(pixelDataGpu),
        0 // unused
    };

    vulkanDevice->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
       vulkanDevice->layout[vulkanDevice->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 2,
        pushConstants
    );

    Allocation indexAlloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(indicesGpu));

    vulkanDevice->dispatchTable.cmdBindIndexBuffer(
        cb->commandBuffer,
        indexAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(indicesGpu) - indexAlloc.address,
        VK_INDEX_TYPE_UINT32
    );

    vulkanDevice->dispatchTable.cmdDrawIndexed(
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
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    VkDeviceAddress pushConstants[3] = {
        reinterpret_cast<VkDeviceAddress>(vertexDataGpu),
        reinterpret_cast<VkDeviceAddress>(pixelDataGpu),
        0 // unused
    };

    vulkanDevice->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
       vulkanDevice->layout[vulkanDevice->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 2,
        pushConstants
    );

    Allocation indexAlloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(indicesGpu));

    vulkanDevice->dispatchTable.cmdBindIndexBuffer(
        cb->commandBuffer,
        indexAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(indicesGpu) - indexAlloc.address,
        VK_INDEX_TYPE_UINT32
    );

    Allocation argsAlloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(argsGpu));

    vulkanDevice->dispatchTable.cmdDrawIndexedIndirect(
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
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    VkDeviceAddress pushConstants[3] = {
        reinterpret_cast<VkDeviceAddress>(dataVxGpu),
        reinterpret_cast<VkDeviceAddress>(dataPxGpu),
        static_cast<VkDeviceAddress>(vxStride | (static_cast<uint64_t>(pxStride) << 32)) // pack strides into a single 64-bit value
    };

    vulkanDevice->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
        vulkanDevice->layout[vulkanDevice->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 3,
        pushConstants
    );

    Allocation indexAlloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(indicesGpu));

    vulkanDevice->dispatchTable.cmdBindIndexBuffer(
        cb->commandBuffer,
        indexAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(indicesGpu) - indexAlloc.address,
        VK_INDEX_TYPE_UINT32
    );

    Allocation argsAlloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(argsGpu));

    Allocation countAlloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(drawCountGpu));

    VkDrawIndexedIndirectCommand drawCommand = {};

    vulkanDevice->dispatchTable.cmdDrawIndexedIndirectCount(
        cb->commandBuffer,
        argsAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(argsGpu) - argsAlloc.address,
        countAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(drawCountGpu) - countAlloc.address,
        vulkanDevice->physicalDeviceProperties2.properties.limits.maxDrawIndirectCount,
        sizeof(VkDrawIndexedIndirectCommand)
    );
}

void gpuDrawMeshlets(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, uint3 dim)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    VkDeviceAddress pushConstants[2] = {
        reinterpret_cast<VkDeviceAddress>(meshletDataGpu),
        reinterpret_cast<VkDeviceAddress>(pixelDataGpu)
    };

    vulkanDevice->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
       vulkanDevice->layout[vulkanDevice->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 2,
        pushConstants
    );
    
    vulkanDevice->dispatchTable.cmdDrawMeshTasksEXT(
        cb->commandBuffer,
        dim.x,
        dim.y,
        dim.z
    );
}

void gpuDrawMeshletsIndirect(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, void *dimGpu)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
    VkDeviceAddress pushConstants[2] = {
        reinterpret_cast<VkDeviceAddress>(meshletDataGpu),
        reinterpret_cast<VkDeviceAddress>(pixelDataGpu)
    };

    vulkanDevice->dispatchTable.cmdPushConstants(
        cb->commandBuffer,
       vulkanDevice->layout[vulkanDevice->currentPipeline[cb]->bindPoint],
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(VkDeviceAddress) * 2,
        pushConstants
    );

    Allocation dimAlloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(dimGpu));

    vulkanDevice->dispatchTable.cmdDrawMeshTasksIndirectEXT(
        cb->commandBuffer,
        dimAlloc.buffer,
        reinterpret_cast<VkDeviceAddress>(dimGpu) - dimAlloc.address,
        1,
        0
    );
}

#ifdef GPU_SURFACE_EXTENSION
Span<FORMAT> gpuSurfaceFormats(GpuDevice device, GpuSurface surface)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    if (surface == nullptr)
    {
        return Span<FORMAT>();
    }

    if (surface->formats.empty())
    {
        uint32_t formatCount = 0;
        vulkanInstance->instanceDispatchTable.getPhysicalDeviceSurfaceFormatsKHR(
            vulkanDevice->physicalDevice, 
            surface->surface, 
            &formatCount, 
            nullptr
        );

        std::vector<VkSurfaceFormatKHR> vkFormats(formatCount);
        vulkanInstance->instanceDispatchTable.getPhysicalDeviceSurfaceFormatsKHR(
            vulkanDevice->physicalDevice, 
            surface->surface, 
            &formatCount, 
            vkFormats.data()
        );

        for (const auto& vkFormat : vkFormats)
        {
            surface->formats.push_back(gpuVkFormatToGpuFormat(vkFormat.format));
        }
    }

    return Span<FORMAT>(surface->formats);
}

GpuSwapchain gpuCreateSwapchain(GpuDevice device, GpuSurface surface, uint32_t images)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    struct GpuSurfaceImpl
    {
        VkSurfaceKHR surface;
    };
    auto surf = reinterpret_cast<GpuSurfaceImpl*>(surface);

    VkSurfaceFormatKHR surfaceFormat;

    auto builder = vkb::SwapchainBuilder{vulkanDevice->device, surf->surface}
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    vulkanDevice->device.surface = surf->surface;
    auto presentQueue = new GpuQueue_T{ vulkanDevice->device.get_queue(vkb::QueueType::present).value(), device };
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
        vulkanDevice->dispatchTable.createImageView(&viewInfo, nullptr, &view);

        swapchainImages.push_back(new GpuTexture_T {
            desc,
            image,
            view,
            device
        });
    }

    std::vector<VkSemaphore> presentSemaphores;

    for (size_t i = 0; i < swapchainImages.size(); i++)
    {
        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkSemaphore presentSemaphore;
        vulkanDevice->dispatchTable.createSemaphore(&semaphoreInfo, nullptr, &presentSemaphore);
        presentSemaphores.push_back(presentSemaphore);
    }

    return new GpuSwapchain_T { 
        swapchain.swapchain, 
        presentQueue->queue, 
        0, 
        desc,
        swapchainImages, 
        presentSemaphores,
        device
    };
}

void gpuDestroySwapchain(GpuSwapchain swapchain)
{
    VulkanDevice* vulkanDevice = swapchain->device->vulkanDevice;
    for (auto image : swapchain->images)
    {
        vulkanDevice->dispatchTable.destroyImageView(image->view, nullptr);
        delete image;
    }
    vulkanDevice->dispatchTable.destroySwapchainKHR(swapchain->swapchain, nullptr);
    for (auto sema : swapchain->presentSemaphores)
    {
        vulkanDevice->dispatchTable.destroySemaphore(sema, nullptr);
    }
    delete swapchain;
}

GpuTextureDesc gpuSwapchainDesc(GpuSwapchain swapchain)
{
    return swapchain->desc;
}

GpuTexture gpuSwapchainImage(GpuSwapchain swapchain)
{
    VulkanDevice* vulkanDevice = swapchain->device->vulkanDevice;
    vulkanDevice->dispatchTable.resetFences(1, &vulkanDevice->acquireFence);

    vulkanDevice->dispatchTable.acquireNextImageKHR(
        swapchain->swapchain,
        UINT64_MAX,
        VK_NULL_HANDLE,
        vulkanDevice->acquireFence,
        &swapchain->imageIndex
    );

    vulkanDevice->dispatchTable.waitForFences(1, &vulkanDevice->acquireFence, VK_TRUE, UINT64_MAX);

    return swapchain->images[swapchain->imageIndex];
}

void gpuPresent(GpuSwapchain swapchain, GpuSemaphore sema, uint64_t value)
{
    VulkanDevice* vulkanDevice = swapchain->device->vulkanDevice;
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
    vulkanDevice->dispatchTable.queueSubmit2(
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

    vulkanDevice->dispatchTable.queuePresentKHR(
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
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
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

GpuAccelerationStructureSizes gpuAccelerationStructureSizes(GpuDevice device, GpuAccelerationStructureDesc desc)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = gpuBuildInfoToVkBuildInfo(desc, geometries);
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

    std::vector<uint32_t> primitiveCounts;
    for (const auto& range : desc.buildRanges)
    {
        primitiveCounts.push_back(range.primitiveCount);
    }

    vulkanDevice->dispatchTable.getAccelerationStructureBuildSizesKHR(
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo,
        primitiveCounts.data(),
        &sizeInfo
    );

    return {
        sizeInfo.accelerationStructureSize,
        sizeInfo.updateScratchSize,
        sizeInfo.buildScratchSize
    };
}

GpuAccelerationStructure gpuCreateAccelerationStructure(GpuDevice device, GpuAccelerationStructureDesc desc, void *ptrGpu, uint64_t size)
{
    VulkanDevice* vulkanDevice = device->vulkanDevice;
    auto alloc = vulkanDevice->findAllocation(reinterpret_cast<VkDeviceAddress>(ptrGpu));

    VkAccelerationStructureCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = alloc.buffer;
    createInfo.offset = reinterpret_cast<VkDeviceAddress>(ptrGpu) - alloc.address;
    createInfo.size = size;
    createInfo.type = (desc.type == TYPE_BOTTOM_LEVEL) ? VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    VkAccelerationStructureKHR vkAs;
    vulkanDevice->dispatchTable.createAccelerationStructureKHR(
        &createInfo,
        nullptr,
        &vkAs
    );

    auto as = new GpuAccelerationStructure_T();
    as->device = device;

    as->buildInfo = gpuBuildInfoToVkBuildInfo(desc, as->geometries);
    as->buildInfo.dstAccelerationStructure = vkAs;

    for (const auto& range : desc.buildRanges)
    {
        VkAccelerationStructureBuildRangeInfoKHR vkRange = {};
        vkRange.firstVertex = range.firstVertex;
        vkRange.primitiveCount = range.primitiveCount;
        vkRange.primitiveOffset = range.primitiveOffset;
        vkRange.transformOffset = range.transformOffset;
        as->buildRanges.push_back(vkRange);
    }

    return as;
}

void gpuBuildAccelerationStructures(GpuCommandBuffer cb, Span<GpuAccelerationStructure> as, void *scratchGpu, MODE mode)
{
    VulkanDevice* vulkanDevice = cb->device->vulkanDevice;
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
        buildRanges.push_back(a->buildRanges.data());
    }

    vulkanDevice->dispatchTable.cmdBuildAccelerationStructuresKHR(
        cb->commandBuffer,
        as.size(),
        buildInfos.data(),
        buildRanges.data()
    );
}

void gpuDestroyAccelerationStructure(GpuAccelerationStructure as)
{
    VulkanDevice* vulkanDevice = as->device->vulkanDevice;
    vulkanDevice->dispatchTable.destroyAccelerationStructureKHR(as->buildInfo.dstAccelerationStructure, nullptr);
    delete as;
}
#endif // GPU_RAY_TRACING_EXTENSION