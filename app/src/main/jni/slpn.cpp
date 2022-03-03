//
// Created by 12108 on 2021/11/29.
//

#include "slpn.h"


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

int k1=0;

std::vector<int> pair_line = {
        0,2,
        2,4,

        5,6,

        6,8,
        8,10,
        6,12,
        12,14,
        14,16,


        0,1,
        1,3,

        11,12,

        5,7,
        7,9,
        5,11,
        11,13,
        13,15,
};
const float mean1[3] = { 103.53, 116.28, 123.675 };
const float std1[3] = { 57.375, 57.12, 58.395 };

/*
mat转换为数组，即指针
*/
void convertMat2pointer(cv::Mat& img, float* x){
    for (int i = 0; i < img.rows; ++i) {

        float* cur_row = img.ptr<float>(i);
        for (int j = 0; j < img.cols; ++j) {

            x[i * img.cols + j] = *cur_row++;
            x[img.rows * img.cols + i * img.cols + j] = *cur_row++;
            x[img.rows * img.cols * 2 + i * img.cols + j] = *cur_row++;
        }
    }
}

/*
图像的预处理，减去mean除以std
*/
void _normalize(cv::Mat& img){
    img.convertTo(img, CV_32F);
    for (int i = 0; i < img.rows; ++i) {
        float* cur_row = img.ptr<float>(i);
        for (int j = 0; j < img.cols; ++j) {
            *cur_row++ = (*cur_row - mean1[0]) / std1[0];
            *cur_row++ = (*cur_row - mean1[1]) / std1[1];
            *cur_row++ = (*cur_row - mean1[2]) / std1[2];
        }
    }
}

void box_to_center_scale(cv::Rect_<float> rect, int width, int height, std::vector<float> &center, std::vector<float> &scale) {
    int box_width = rect.width;
    int box_height = rect.height;
    center[0] = rect.x + box_width * 0.5;
    center[1] = rect.y + box_height * 0.5;
    float aspect_ratio = width * 1.0 / height;
    int pixel_std = 200;
    if (box_width > aspect_ratio * box_height) {
        box_height = box_width * 1.0 / aspect_ratio;
    }
    else if (box_width < aspect_ratio * box_height) {
        box_width = box_height * aspect_ratio;
    }
    scale[0] = box_width * 1.0 / pixel_std;
    scale[1] = box_height * 1.0 / pixel_std;
    if (center[0] != -1) {
        scale[0] = scale[0] * 1.25;
        scale[1] = scale[1] * 1.25;
    }
}

std::vector<float> get_3rd_point(std::vector<float>& a, std::vector<float>& b) {
    std::vector<float> direct{ a[0] - b[0],a[1] - b[1] };
    return std::vector<float>{b[0] - direct[1], b[1] + direct[0]};
}

std::vector<float> get_dir(float src_point_x,float src_point_y,float rot_rad){
    float sn = sin(rot_rad);
    float cs = cos(rot_rad);
    std::vector<float> src_result{ 0.0,0.0 };
    src_result[0] = src_point_x * cs - src_point_y * sn;
    src_result[1] = src_point_x * sn + src_point_y * cs;
    return src_result;
}

void affine_tranform(float pt_x, float pt_y, cv::Mat& t,float* x,int p,int num) {
    float new1[3] = { pt_x, pt_y, 1.0 };
    cv::Mat new_pt(3, 1, t.type(), new1);
    cv::Mat w = t * new_pt;
    x[p] = w.at<float>(0,0);
    x[p + num] = w.at<float>(0,1);
}

cv::Mat get_affine_transform(std::vector<float>& center, std::vector<float>& scale, float rot, std::vector<int>& output_size,int inv) {
    std::vector<float> scale_tmp;
    scale_tmp.push_back(scale[0] * 200);
    scale_tmp.push_back(scale[1] * 200);
    float src_w = scale_tmp[0];
    int dst_w = output_size[0];
    int dst_h = output_size[1];
    float rot_rad = rot * 3.1415926535 / 180;
    std::vector<float> src_dir = get_dir(0, -0.5 * src_w, rot_rad);
    std::vector<float> dst_dir{ 0.0, float(-0.5) * dst_w };
    std::vector<float> src1{ center[0] + src_dir[0],center[1] + src_dir[1] };
    std::vector<float> dst0{ float(dst_w * 0.5),float(dst_h * 0.5) };
    std::vector<float> dst1{ float(dst_w * 0.5) + dst_dir[0],float(dst_h * 0.5) + dst_dir[1] };
    std::vector<float> src2 = get_3rd_point(center, src1);
    std::vector<float> dst2 = get_3rd_point(dst0, dst1);
    if (inv == 0) {
        float a[6][6] = { {center[0],center[1],1,0,0,0},
                          {0,0,0,center[0],center[1],1},
                          {src1[0],src1[1],1,0,0,0},
                          {0,0,0,src1[0],src1[1],1},
                          {src2[0],src2[1],1,0,0,0},
                          {0,0,0,src2[0],src2[1],1} };
        float b[6] = { dst0[0],dst0[1],dst1[0],dst1[1],dst2[0],dst2[1] };
        cv::Mat a_1 = cv::Mat(6, 6, CV_32F, a);
        cv::Mat b_1 = cv::Mat(6, 1, CV_32F, b);
        cv::Mat result;
        solve(a_1, b_1, result, 0);
        cv::Mat dst = result.reshape(0, 2);
        return dst;
    }
    else {
        float a[6][6] = { {dst0[0],dst0[1],1,0,0,0},
                          {0,0,0,dst0[0],dst0[1],1},
                          {dst1[0],dst1[1],1,0,0,0},
                          {0,0,0,dst1[0],dst1[1],1},
                          {dst2[0],dst2[1],1,0,0,0},
                          {0,0,0,dst2[0],dst2[1],1} };
        float b[6] = { center[0],center[1],src1[0],src1[1],src2[0],src2[1] };
        cv::Mat a_1 = cv::Mat(6, 6, CV_32F, a);
        cv::Mat b_1 = cv::Mat(6, 1, CV_32F, b);
        cv::Mat result;
        solve(a_1, b_1, result, 0);
        cv::Mat dst = result.reshape(0, 2);
        return dst;
    }
}

/*
* 该函数暂时只实现了batch为1的情况
*/
void get_max_preds(ncnn::Mat out, float* preds, float* maxvals) {
    int batch_size = 1;
    int num_joints = out.c;
    int width = out.w;
    float* pred_mask = new float[num_joints * 2];
    int* idx = new int[num_joints * 2];
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_joints; ++j) {
            float max = out[i * num_joints * out.h * out.w + j * out.h * out.w];
            int max_id = 0;
            for (int k = 1; k < out.h * out.w; ++k) {
                int index = i * num_joints * out.h * out.w + j * out.h * out.w + k;
                if (out[index] > max) {
                    max = out[index];
                    max_id = k;
                }
            }
            maxvals[j] = max;
            idx[j] = max_id;
            idx[j + num_joints] = max_id;
        }
    }
    for (int i = 0; i < num_joints; ++i) {
        idx[i] = idx[i] % width;
        idx[i + num_joints] = idx[i + num_joints] / width;
        if (maxvals[i] > 0) {
            pred_mask[i] = 1.0;
            pred_mask[i + num_joints] = 1.0;
        }
        else {
            pred_mask[i] = 0.0;
            pred_mask[i + num_joints] = 0.0;
        }
        preds[i] = idx[i] * pred_mask[i];
        preds[i + num_joints] = idx[i + num_joints] * pred_mask[i + num_joints];
    }

}

/*
得到数字的符号
*/
int sign(float x) {
    int w = 0;
    if (x > 0) {
        w = 1;
    }
    else if (x == 0) {
        w = 0;
    }
    else {
        w = -1;
    }
    return w;
}

void transform_preds(float* coords, std::vector<float>& center, std::vector<float>& scale, std::vector<int>& output_size, int64_t t, float* target_coords) {
    cv::Mat tran = get_affine_transform(center, scale, 0, output_size, 1);
    for (int p = 0; p < t; ++p) {
        affine_tranform(coords[p], coords[p + t], tran, target_coords, p, t);
    }
}

void get_final_preds(ncnn::Mat out, std::vector<float>& center, std::vector<float> scale, float* preds) {
    float* coords = new float[out.c * 2];
    float* maxvals = new float[out.c];
    int heatmap_height = out.h;
    int heatmap_width = out.w;
    get_max_preds(out, coords, maxvals);
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < out.c; ++j) {
            int px = int(coords[i * out.c + j] + 0.5);
            int py = int(coords[i * out.c + j + out.c] + 0.5);
            int index = (i * out.c + j) * out.h * out.w;
            if (px > 1 && px < heatmap_width - 1 && py>1 && py < heatmap_height - 1) {
                float diff_x = out[index + py * heatmap_width + px + 1] - out[index + py * heatmap_width + px - 1];
                float diff_y = out[index + (py + 1) * heatmap_width + px] - out[index + (py - 1) * heatmap_width+ px];
                coords[i * out.c + j] += sign(diff_x) * 0.25;
                coords[i * out.c + j + out.c] += sign(diff_y) * 0.25;
            }
        }
    }
    std::vector<int> img_size{ heatmap_width,heatmap_height };
    transform_preds(coords, center, scale, img_size, out.c, preds);
}

//void convertMat2pointer(cv::Mat& img, float* x){
//    for (int i = 0; i < img.rows; ++i) {
//
//        float* cur_row = img.ptr<float>(i);
//        for (int j = 0; j < img.cols; ++j) {
//
//            x[i * img.cols + j] = *cur_row++;
//            x[img.rows * img.cols + i * img.cols + j] = *cur_row++;
//            x[img.rows * img.cols * 2 + i * img.cols + j] = *cur_row++;
//        }
//    }
//}


SLPNet::SLPNet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int SLPNet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    slpnet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    slpnet.opt = ncnn::Option();

#if NCNN_VULKAN
    slpnet.opt.use_vulkan_compute = use_gpu;
#endif
//    slpnet.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    slpnet.opt.num_threads = ncnn::get_big_cpu_count();
    slpnet.opt.blob_allocator = &blob_pool_allocator;
    slpnet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    slpnet.load_param(parampath);
    slpnet.load_model(modelpath);


    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int SLPNet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    slpnet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    slpnet.opt = ncnn::Option();
#if NCNN_VULKAN
    slpnet.opt.use_vulkan_compute = use_gpu;
#endif
//    slpnet.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    slpnet.opt.num_threads = ncnn::get_big_cpu_count();
    slpnet.opt.blob_allocator = &blob_pool_allocator;
    slpnet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    __android_log_print(ANDROID_LOG_DEBUG, "model", "%s", modeltype);
    slpnet.load_param(mgr, parampath);
    slpnet.load_model(mgr, modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int SLPNet::detect(const cv::Mat& rgb, cv::Rect_<float> rect)
{
    std::vector<float> center{0.0,0.0};
    std::vector<float> scale{0.0,0.0};
    box_to_center_scale(rect, img_size[0], img_size[1], center, scale);

    cv::Mat input;

    cv::Mat tran = get_affine_transform(center, scale, 0, img_size,0);

    cv::warpAffine(rgb, input, tran, cv::Size(img_size[0],img_size[1]), cv::INTER_LINEAR);

    _normalize(input);

//    cv::imwrite("/storage/emulated/0/abcd/" + std::to_string(k1) + ".jpg", input);
    //k1++;
    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(input.data, ncnn::Mat::PIXEL_RGB, 192, 256, 192, 256);


    //ncnn::Mat in = ncnn::Mat::from_pixels(input.data, ncnn::Mat::PIXEL_RGB, 192, 256);

    convertMat2pointer(input, x);

    ncnn::Mat in(192, 256, 3, x);

//    float* a = new float[11*10*17];
//    for(int j=0;j<11*10*17;++j){
//        a[j]=float(j);
//    }
//    ncnn::Mat a1(11,10,17,a);
//    char text[256];
//    sprintf(text, "%d %d %d %d %.0f %.0f %.2f %.2f %d %d", a1.dims, a1.h, a1.w, a1.c,a1.channel(0).row(10)[0],
//            a[0*11*10+10*10], center[0], scale[0], img_w, img_h);
//    cv::Scalar textcc = cv::Scalar(255, 255, 255);
//    cv::putText(rgb, text, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);

    // so for 0-255 input image, rgb_mean should multiply 255 and norm should div by std.
    //in.substract_mean_normalize(mean_vals, norm_vals);


    ncnn:: Extractor ex = slpnet.create_extractor();

    ex.input("data", in);
//    ex.input("input.1", in_pad);



    {
        ncnn::Mat out;
//        __android_log_print(ANDROID_LOG_DEBUG, "model", "sssss222");
        ex.extract("output_heatmaps", out);
//        char text[256];
//        sprintf(text, "%d %d %d %d %f %f", out.dims, out.h, out.w, out.c,
//                out.channel(2).row(10)[0], out[2*out.h*out.w+10*out.w]);
//        cv::Scalar textcc = cv::Scalar(255, 255, 255);
//        cv::putText(rgb, text, cv::Point(100, 200), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
//
//
//        sprintf(text, "%d %d %d %d %f %f", in.dims, in.h, in.w, in.c,
//                in.channel(2).row(10)[0], in[0]);
//        textcc = cv::Scalar(255, 255, 255);
//        cv::putText(rgb, text, cv::Point(20, 300), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);


        float* preds = new float[out.c*2+2];
        get_final_preds(out, center, scale, preds);

        preds[34] = (preds[5] + preds[6]) / 2;
        preds[35] = (preds[5+17] + preds[+17]) / 2;
        int line_begin, line_end, iter, point_begin;
        if (1) {//所有关节点
            line_begin = 0;
            line_end = pair_line.size();
            iter = 1;
            point_begin = 0;
        }
        for (int i = line_begin; i < line_end; i = i + 2) {
            cv::line(rgb, cv::Point2d(int(preds[pair_line[i]]), int(preds[pair_line[i] + 17])),
                     cv::Point2d(int(preds[pair_line[i + 1]]), int(preds[pair_line[i + 1] + 17])), cv::Scalar(0, 255, 0), 4);
        }

        for (int i = point_begin; i < 17; i=i+iter) {
            int x_coord = int(preds[i]);
            int y_coord = int(preds[i + 17]);
            cv::circle(rgb, cv::Point2d(x_coord, y_coord), 1, (0, 255, 0), 2);
            // cv::putText(rgb, std::to_string(i), cv::Point2d(x_coord, y_coord), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
//            __android_log_print(ANDROID_LOG_DEBUG, "%s", std::to_string(i).c_str(), "%s",std::to_string(x_coord).c_str());
        }


//        int a = ex.extract("10128", out);

//        __android_log_print(ANDROID_LOG_DEBUG, "model", "sssss");

//        char text[256];
//        sprintf(text, "%d %d %d %d%%", out.dims, out.h, out.w, out.c);
//        cv::Scalar textcc = cv::Scalar(255, 255, 255);
//        cv::putText(rgb, text, cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);



//        std::vector<int> strides = {8, 16, 32}; // might have stride=64
//        std::vector<GridAndStride> grid_strides;
//        generate_grids_and_stride(target_size, strides, grid_strides);
//        generate_yolox_proposals(grid_strides, out, prob_threshold, proposals);
    }



    // sort all proposals by score from highest to lowest
//    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
//    std::vector<int> picked;
//    nms_sorted_bboxes(proposals, picked, nms_threshold);

//    int count = picked.size();

//    objects.resize(count);
//    for (int i = 0; i < count; i++)
//    {
//        objects[i] = proposals[picked[i]];
//
//        // adjust offset to original unpadded
//        float x0 = (objects[i].rect.x) / scale;
//        float y0 = (objects[i].rect.y) / scale;
//        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
//        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;
//
//        // clip
//        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
//        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
//        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
//        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);
//
//        objects[i].rect.x = x0;
//        objects[i].rect.y = y0;
//        objects[i].rect.width = x1 - x0;
//        objects[i].rect.height = y1 - y0;
//    }

    return 0;
}

int SLPNet::draw(cv::Mat& rgb)
{


    return 0;
}