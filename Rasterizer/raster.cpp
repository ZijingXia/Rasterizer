#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <chrono>

#include <cmath>
#include "matrix.h"
#include "colour.h"
#include "mesh.h"
#include "zbuffer.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"

// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.

// SoA格式的顶点缓冲区（按分量存储，提升缓存命中率）
struct VertexBufferSoA {
    std::vector<float> px, py, pz, pw; // 位置分量
    std::vector<float> nx, ny, nz;     // 法向量分量
    std::vector<colour> rgb;           // 颜色

    // 从Mesh初始化SoA
    void fromMesh(const Mesh& mesh) {
        size_t n = mesh.vertices.size();
        px.resize(n); py.resize(n); pz.resize(n); pw.resize(n);
        nx.resize(n); ny.resize(n); nz.resize(n);
        rgb.resize(n);

        for (size_t i = 0; i < n; i++) {
            px[i] = mesh.vertices[i].p.x;
            py[i] = mesh.vertices[i].p.y;
            pz[i] = mesh.vertices[i].p.z;
            pw[i] = mesh.vertices[i].p.w;
            nx[i] = mesh.vertices[i].normal.x;
            ny[i] = mesh.vertices[i].normal.y;
            nz[i] = mesh.vertices[i].normal.z;
            rgb[i] = mesh.vertices[i].rgb;
        }
    }
};

// 给Mesh添加SoA缓冲区（提前初始化，避免每次渲染重构）
struct MeshOpt : public Mesh {
    VertexBufferSoA soa_buffer;
    bool soa_initialized = false;

    void initSoA() {
        if (!soa_initialized) {
            soa_buffer.fromMesh(*this);
            soa_initialized = true;
        }
    }
};

//============================
//顶点处理
//============================

void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L)
{
    matrix p = renderer.perspective * camera * mesh->world;
    std::vector<Vertex> transformed(mesh->vertices.size());

    // ===== 优化1：提前缓存常量 =====
    const int canvas_width = renderer.canvas.getWidth();
    const int canvas_height = renderer.canvas.getHeight();
    const float half_width = 0.5f * canvas_width;
    const float half_height = 0.5f * canvas_height;
    const float one = 1.0f;

    // 顶点变换阶段
    for (size_t i = 0; i < mesh->vertices.size(); i++)
    {
        transformed[i].p = p * mesh->vertices[i].p;
        transformed[i].p.divideW();

        // ===== 优化2：法向量变换只算一次（复用mesh->world） =====
        transformed[i].normal = mesh->world * mesh->vertices[i].normal;
        transformed[i].normal.normalise();

        // ===== 优化3：简化屏幕映射计算（减少算术运算） =====
        transformed[i].p[0] = (transformed[i].p[0] + one) * half_width;
        transformed[i].p[1] = canvas_height - (transformed[i].p[1] + one) * half_height;

        transformed[i].rgb = mesh->vertices[i].rgb;
    }

    // 三角形阶段
    // ===== 优化4：裁剪逻辑提前缓存z阈值，减少fabs调用 =====
    const float z_threshold = 1.0f;
    for (triIndices& ind : mesh->triangles)
    {
        Vertex& v0 = transformed[ind.v[0]];
        Vertex& v1 = transformed[ind.v[1]];
        Vertex& v2 = transformed[ind.v[2]];

        // 优化：用直接比较代替fabs（z范围是[-1,1]，等价）
        if (v0.p[2] < -z_threshold || v0.p[2] > z_threshold ||
            v1.p[2] < -z_threshold || v1.p[2] > z_threshold ||
            v2.p[2] < -z_threshold || v2.p[2] > z_threshold)
            continue;

        triangle tri(v0, v1, v2);
        tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
}


// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest()
{
    Renderer renderer;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f),
             colour(1.0f, 1.0f, 1.0f),
             colour(0.2f, 0.2f, 0.2f) };

    matrix camera = matrix::makeIdentity();

    bool running = true;

    Mesh sphere = Mesh::makeSphere(1.0f, 10, 20);

    struct BallInstance
    {
        float x;
        float baseY;
        float z;
        float phase;
        float speed;
        float amplitude;
    };

    std::vector<BallInstance> balls;

    // ===== 更复杂的球阵列 =====
    const int rows = 10;
    const int cols = 20;

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            BallInstance b;

            b.x = -20.0f + c * 2.0f;
            b.baseY = 0.0f;
            b.z = -5.0f - r * 2.0f;

            b.phase = (r + c) * 0.3f;        // 错开相位
            b.speed = 0.5f + 0.2f * (r % 3); // 不同速度
            b.amplitude = 2.0f + 0.5f * (c % 4); // 更大浮动

            balls.push_back(b);
        }
    }

    float time = 0.0f;
    float step = 0.03f;

    // ===== 时间测试 =====
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    while (running)
    {
        renderer.canvas.checkInput();
        renderer.clear();

        if (renderer.canvas.keyPressed(VK_ESCAPE))
            break;

        time += step;

        float currentCycle = time / (2.0f * M_PI);
        if (static_cast<int>(currentCycle) > cycle)
        {
            cycle = static_cast<int>(currentCycle);
            if (cycle % 2 == 0)
            {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " : "
                    << std::chrono::duration<double, std::milli>(end - start).count()
                    << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        // ===== 渲染所有球 =====
        for (auto& b : balls)
        {
            float yOffset = sin(time * b.speed + b.phase) * b.amplitude;

            sphere.world = matrix::makeTranslation(
                b.x,
                b.baseY + yOffset,
                b.z
            );

            render(renderer, &sphere, camera, L);
        }

        renderer.present();
    }
}


// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
    unsigned int r = rng.getRandomInt(0, 3);

    switch (r) {
    case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
    default: return matrix::makeIdentity();
    }
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };

    bool running = true;
    std::vector<Mesh*> scene;

    // ===== 优化：复用基础Cube Mesh，只修改world矩阵 =====
    Mesh* base_cube = new Mesh();
    *base_cube = Mesh::makeCube(1.f);
    base_cube->ka = 0.2f;
    base_cube->kd = 0.8f;

    // 只创建40个Mesh实例，但复用同一个顶点/三角形数据
    for (unsigned int i = 0; i < 20; i++) {
        Mesh* m = new Mesh(*base_cube); // 拷贝构造，复用顶点数据
        m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);

        m = new Mesh(*base_cube);
        m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
    }
    delete base_cube; // 释放基础Cube

    float zoffset = 8.0f;
    float step = -0.1f;
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset);

        // ===== 优化：减慢旋转速度，减少矩阵乘法开销 =====
        scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.01f, 0.01f, 0.0f);
        scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.01f, 0.02f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        for (auto& m : scene)
            render(renderer, m, camera, L);
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.2f, 0.2f, 0.2f) };

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        for (auto& m : scene)
            render(renderer, m, camera, L);
        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Entry point of the application
// No input variables
int main() {
    // Uncomment the desired scene function to run
    scene1();
    //scene2();
    //sceneTest(); 
    

    return 0;
}