//#include <QGuiApplication>
//#include <QtGlobal>
//#include <QVulkanInstance>
#include <iostream>
#include <random>
//#include <QLoggingCategory>

//#include "Window.h"
#include "GLFWVulkanWindow.h"
#include "Renderer.h"

#include "gpuHashTable/linearprobing.h"

int main(int argc, char *argv[]) {
    /*
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(1,2000000);

    std::vector<int> keys;
    int tableSize = 1024 * 1024;
    for (int i = 0; i < tableSize; i++) {
        keys.push_back(dist6(rng));
    }

    std::vector<char> inserted(keys.size());
    int* hashTable = create_hashtable(tableSize);
    insert_hashtable(hashTable, keys.data(), keys.size(), inserted.data());

    hashTable = resize(hashTable, 1024 * 1024 * 4);
    */


    /*
    std::random_device rd;
    uint32_t seed = rd();
    std::mt19937 rnd(seed);  // mersenne_twister_engine

    int kNumKeyValues = 5;
    //std::vector<KeyValue> insert_kvs = generate_random_keyvalues(rnd, kNumKeyValues);


    std::vector<char> inserted(6);

    std::vector<KeyValue> insert_kvs;
    insert_kvs.push_back({0, 9});
    insert_kvs.push_back({2, 8});
    insert_kvs.push_back({4, 7});
    insert_kvs.push_back({6, 7});
    insert_kvs.push_back({8, 7});

    std::vector<KeyValue> delete_kvs = shuffle_keyvalues(rnd, insert_kvs, kNumKeyValues / 2);

    KeyValue* pHashTable = create_hashtable();

    // Insert items into the hash table
    const uint32_t num_insert_batches = 1;
    uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
    for (uint32_t i = 0; i < num_insert_batches; i++)
    {
        insert_hashtable(pHashTable, insert_kvs.data() + i * num_inserts_per_batch, num_inserts_per_batch, inserted.data());
    }
    for (char c : inserted) {
        std::cout << int(c) << std::endl;
    }
    for (int i = 0; i < inserted.size(); i++) {
        inserted[i] = 0;
    }

    std::vector<KeyValue> lookup;
    lookup.push_back({2, 0});
    lookup.push_back({4, 0});
    lookup.push_back({8, 0});
    lookup.push_back({6, 0});
    lookup.push_back({0, 0});
    lookup_hashtable(pHashTable, lookup.data(), num_inserts_per_batch);
    for (KeyValue kv : lookup) {
        std::cout << kv.key << " " << kv.value << std::endl;
    }

    for (int i = 0; i < inserted.size(); i++) {
        inserted[i] = 0;
    }
    insert_kvs.clear();
    insert_kvs.push_back({2, 80});
    insert_kvs.push_back({4, 7});
    insert_kvs.push_back({6, 7});
    insert_kvs.push_back({8, 7});
    num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
    insert_hashtable(pHashTable, insert_kvs.data(), num_inserts_per_batch, inserted.data());
    for (char c : inserted) {
        std::cout << int(c) << std::endl;
    }


    for (int i = 0; i < inserted.size(); i++) {
        inserted[i] = 0;
    }
    insert_kvs.push_back({5, 7});
    insert_kvs.push_back({333, 90});
    num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
    for (uint32_t i = 0; i < num_insert_batches; i++)
    {
        insert_hashtable(pHashTable, insert_kvs.data() + i * num_inserts_per_batch, num_inserts_per_batch, inserted.data());
    }
    for (char c : inserted) {
        std::cout << int(c) << std::endl;
    }
    lookup.clear();
    lookup.push_back({2, 0});
    lookup.push_back({4, 0});
    lookup.push_back({8, 0});
    lookup.push_back({6, 0});
    lookup.push_back({0, 0});
    lookup.push_back({5, 0});
    lookup.push_back({333, 0});
    num_inserts_per_batch = (uint32_t)lookup.size() / num_insert_batches;
    lookup_hashtable(pHashTable, lookup.data(), num_inserts_per_batch);
    for (KeyValue kv : lookup) {
        std::cout << kv.key << " " << kv.value << std::endl;
    }


    for (uint32_t i = 0; i < num_insert_batches; i++)
    {
        lookup_hashtable(pHashTable, insert_kvs.data() + i * num_inserts_per_batch, num_inserts_per_batch);
    }

    // Delete items from the hash table
    const uint32_t num_delete_batches = 1;
    uint32_t num_deletes_per_batch = (uint32_t)delete_kvs.size() / num_delete_batches;
    for (uint32_t i = 0; i < num_delete_batches; i++)
    {
        delete_hashtable(pHashTable, delete_kvs.data() + i * num_deletes_per_batch, num_deletes_per_batch);
    }

    // Get all the key-values from the hash table
    std::vector<KeyValue> kvs = iterate_hashtable(pHashTable);

    destroy_hashtable(pHashTable);
    */


    /*
    QGuiApplication app(argc, argv);

    QVulkanInstance vulkanInstance;

    // Linux
    //if (qEnvironmentVariableIntValue("QT_VK_DEBUG")) {
        //QLoggingCategory::setFilterRules(QStringLiteral("qt.vulkan=true"));
        //vulkanInstance.setLayers(QByteArrayList() << "VK_LAYER_LUNARG_standard_validation"");
    //}
    vulkanInstance.setLayers(QByteArrayList() << "VK_LAYER_KHRONOS_validation");    // TODO: Remove for release build

    if (!vulkanInstance.create()) {
        qWarning("failed to create Vulkan instance");
    }

    Window window;
    window.setDeviceExtensions(
        QByteArrayList()
        << VK_NV_RAY_TRACING_EXTENSION_NAME
        << VK_KHR_MAINTENANCE3_EXTENSION_NAME
        << VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME
    );
    window.setVulkanInstance(&vulkanInstance);
    window.resize(800, 600);
    window.show();

    //VkPhysicalDeviceFeatures f = {};
    //f.samplerAnisotropy = VK_TRUE;

    return app.exec();
    */

    GLFWVulkanWindow app;

    try {
        app.initWindow();
        app.initVulkan();
        app.initRenderer();
        app.mainLoop();
    } catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
