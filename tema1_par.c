#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT 16
#define FILENAME_MAX_SIZE 50
#define STEP 8
#define SIGMA 200
#define RESCALE_X 2048
#define RESCALE_Y 2048

#define CLAMP(v, min, max) \
    if (v < min)           \
    {                      \
        v = min;           \
    }                      \
    else if (v > max)      \
    {                      \
        v = max;           \
    }

typedef struct
{
    ppm_image *image;
    int start_x;
    int end_x;
    ppm_image *new_image;
    int num_threads;
    int thread_index;
    pthread_barrier_t *barrier;
} thread_data_t;

ppm_image **init_contour_map()
{
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++)
    {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

unsigned char **sample_grid(ppm_image *image, int step_x, int step_y, unsigned char sigma)
{
    int p = image->x / step_x;
    int q = image->y / step_y;

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char *));
    if (!grid)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++)
    {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i])
        {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    for (int i = 0; i < p; i++)
    {
        for (int j = 0; j < q; j++)
        {
            ppm_pixel curr_pixel = image->data[i * step_x * image->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma)
            {
                grid[i][j] = 0;
            }
            else
            {
                grid[i][j] = 1;
            }
        }
    }
    grid[p][q] = 0;

    for (int i = 0; i < p; i++)
    {
        ppm_pixel curr_pixel = image->data[i * step_x * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma)
        {
            grid[i][q] = 0;
        }
        else
        {
            grid[i][q] = 1;
        }
    }
    for (int j = 0; j < q; j++)
    {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma)
        {
            grid[p][j] = 0;
        }
        else
        {
            grid[p][j] = 1;
        }
    }

    return grid;
}

void update_image(ppm_image *image, ppm_image *contour, int x, int y)
{
    for (int i = 0; i < contour->x && x + i < image->x; i++)
    {
        for (int j = 0; j < contour->y && y + j < image->y; j++)
        {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            if (image_pixel_index < image->x * image->y)
            {
                image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
                image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
                image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
            }
        }
    }
}

void march(ppm_image *image, unsigned char **grid, ppm_image **contour_map, int step_x, int step_y)
{
    int p = image->x / step_x;
    int q = image->y / step_y;

    for (int i = 0; i < p; i++)
    {
        for (int j = 0; j < q; j++)
        {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(image, contour_map[k], i * step_x, j * step_y);
        }
    }
}

void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x)
{
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++)
    {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++)
    {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

void rescale_image(ppm_image *image, ppm_image *new_image, int num_threads, int thread_index)
{
    uint8_t sample[3];
    // vector pentru valorile rgb ale pixelului

    // daca imaginea originala este mai mica sau egala cu valorile date pentru
    // redimensionare atunci nu se va face nimic si se va copia imaginea originala
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y)
    {
        new_image->data = image->data;
        return;
    }

    // in caz contrar, se va face redimensionarea imaginii
    // intre valorile start si end pentru fiecare thread
    // se va apela functia sample_bicubic pentru a calcula
    // valorile rgb ale pixelului din noua imagine

    else
    {

        int startt_x = thread_index * RESCALE_X / num_threads;
        int temp = (thread_index + 1) * RESCALE_X / num_threads;
        int endd_x = (temp < RESCALE_X) ? temp : RESCALE_X;
        for (int i = startt_x; i < endd_x; i++)
        {
            int j = 0;
            while (j < new_image->y)
            {
                float u = (float)i / (float)(new_image->x - 1);
                float v = (float)j / (float)(new_image->y - 1);
                sample_bicubic(image, u, v, sample);

                new_image->data[i * new_image->y + j].red = sample[0];
                new_image->data[i * new_image->y + j].green = sample[1];
                new_image->data[i * new_image->y + j].blue = sample[2];
                j++;
            }
        }
    }
}

// functia care va fi executata 1 data pentru fiecare thread creat
// si va apela rescale_image pentru a redimensiona imaginea de atatea
// ori cate thread uri sunt. dupa acestea se va folosi o bariera pentru
// sincronizarea thread urilor

void *process_image(void *arg)
{
    thread_data_t *data = (thread_data_t *)arg;

    rescale_image(data->image, data->new_image, data->num_threads, data->thread_index);
    pthread_barrier_wait(data->barrier);
    pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: ./tema1_par <in_file> <out_file> <P>\n");
        return 1;
    }

    // image va fi imaginea initiala, iar new_image va fi imaginea redimensionata
    // new_image va fi folosita pentru a crea harta de contur si va fi mereu scalata
    // la RESCALE_X x RESCALE_Y adica 2048 x 2048. Alocarea se va realiza in main pentru
    // a putea evita problemele ce ar putea aparea de la alocarea dinamica in thread uri

    ppm_image *image = read_ppm(argv[1]);
    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
    new_image->x = RESCALE_X;
    new_image->y = RESCALE_Y;
    new_image->data = (ppm_pixel *)malloc(RESCALE_X * RESCALE_Y * sizeof(ppm_pixel));
    int step_x = STEP;
    int step_y = STEP;
    ppm_image **contour_map = init_contour_map();

    // creeaza p thread uri si le trimite ca argumente imaginea initiala, cea redimensionata,
    // numarul de thread uri, indexul thread ului si bariera pentru sincronizare
    int P = atoi(argv[3]);
    pthread_t threads[P];
    thread_data_t thread_data[P];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, P);
    int i = 0;
    while (i < P)
    {
        thread_data[i].image = image;
        thread_data[i].new_image = new_image;
        thread_data[i].num_threads = P;
        thread_data[i].thread_index = i;
        thread_data[i].barrier = &barrier;
        pthread_create(&threads[i], NULL, process_image, &thread_data[i]);
        i++;
    }

    i = 0;
    while (i < P)
    {
        pthread_join(threads[i], NULL);
        i++;
    }
    // realizeaza harta de contur si o scrie in fisier dupa care se
    // utilizeaza algorithmul march pentru a crea imaginea finala.

    unsigned char **grid = sample_grid(new_image, step_x, step_y, SIGMA);
    march(new_image, grid, contour_map, step_x, step_y);
    // eliberam memoria si pentru imaginea initiala si pentru cea redimensionata
    // si scriem imaginea finala in fisier

    write_ppm(new_image, argv[2]);
    free_resources(new_image, contour_map, grid, step_x);

    pthread_barrier_destroy(&barrier);

    return 0;
}