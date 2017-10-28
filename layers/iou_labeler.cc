#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"

#include <vector>
#include <utility>


// #define DEBUG_LOOP

#ifdef DEBUG_LOOP
#include <iostream>
#endif

using namespace tensorflow;

REGISTER_OP("IouLabeler")
	.Input("bbox_in : float")
	.Input("gt_in : float")
	.Attr("iou_threshold_neg : float")
	.Attr("iou_threshold_pos : float")
	.Output("labeled_bbox : float");

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
		const Tensor& boxes, const Tensor& gt) {
	OP_REQUIRES(context, boxes.dims() == 2,
			errors::InvalidArgument("Boxes must be 2-d",
				boxes.shape().DebugString()));
	OP_REQUIRES(context, boxes.dim_size(1) == 4,
			errors::InvalidArgument("Boxes must have 4 columns"));
	OP_REQUIRES(context, gt.dims() == 2,
			errors::InvalidArgument("Ground truth must be 2-d",
				gt.shape().DebugString()));
	OP_REQUIRES(context, gt.dim_size(1) == 4,
			errors::InvalidArgument("Ground truth must have 4 columns"));
}

static inline float ComputeIOU(typename TTypes<float, 2>::ConstTensor boxes,
		typename TTypes<float, 2>::ConstTensor gt, int i, int j) {
	const float ymin_box = std::min<float>(boxes(i, 0), boxes(i, 2));
	const float xmin_box = std::min<float>(boxes(i, 1), boxes(i, 3));
 	const float ymax_box = std::max<float>(boxes(i, 0), boxes(i, 2));
 	const float xmax_box = std::max<float>(boxes(i, 1), boxes(i, 3));
 	const float ymin_gt = std::min<float>(gt(j, 0), gt(j, 2));
 	const float xmin_gt = std::min<float>(gt(j, 1), gt(j, 3));
 	const float ymax_gt = std::max<float>(gt(j, 0), gt(j, 2));
 	const float xmax_gt = std::max<float>(gt(j, 1), gt(j, 3));

 	// Use modified formula for area calculations to match Faster R-CNN 
 	const float area_box = (ymax_box - ymin_box + 1.0) * (xmax_box - xmin_box + 1.0);
 	const float area_gt = (ymax_gt - ymin_gt + 1.0) * (xmax_gt - xmin_gt + 1.0);
  	if (area_box <= 0 || area_gt <= 0) return 0.0;
 	const float intersection_ymin = std::max<float>(ymin_box, ymin_gt);
 	const float intersection_xmin = std::max<float>(xmin_box, xmin_gt);
 	const float intersection_ymax = std::min<float>(ymax_box, ymax_gt);
 	const float intersection_xmax = std::min<float>(xmax_box, xmax_gt);

 	// Use modified formula for area calculations to match Faster R-CNN
  	const float intersection_area =
     	std::max<float>(intersection_ymax - intersection_ymin + 1.0, 0.0) *
     	std::max<float>(intersection_xmax - intersection_xmin + 1.0, 0.0);
  	return intersection_area / (area_box + area_gt - intersection_area);
}

class IouLabelerOp : public OpKernel {
	public:
		explicit IouLabelerOp(OpKernelConstruction* context)
			: OpKernel(context) { 
			OP_REQUIRES_OK(context, context->GetAttr("iou_threshold_pos", &iou_threshold_pos));
			OP_REQUIRES_OK(context, context->GetAttr("iou_threshold_neg", &iou_threshold_neg));
		}

		void Compute(OpKernelContext* context) override {
			OP_REQUIRES(context, iou_threshold_pos >= 0 && iou_threshold_pos <= 1,
					errors::InvalidArgument("iou_threshold_pos must be in interval [0,1]"));
			OP_REQUIRES(context, iou_threshold_neg >= 0 && iou_threshold_neg <= 1,
					errors::InvalidArgument("iou_threshold_neg must be in interval [0,1]"));
			OP_REQUIRES(context, iou_threshold_pos >= iou_threshold_neg,
					errors::InvalidArgument("iou_threshold_pos must be greater than or"
							" equal to iou_threshold_neg"));

			const Tensor& _boxes = context->input(0);
			const Tensor& _gt = context->input(1);

			ParseAndCheckBoxSizes(context, _boxes, _gt);
			if (!context->status().ok()) return;

			typename TTypes<float, 2>::ConstTensor boxes = _boxes.tensor<float, 2>();
			typename TTypes<float, 2>::ConstTensor gt = _gt.tensor<float, 2>();

			const int num_boxes = _boxes.dim_size(0);
			const int num_gts = _gt.dim_size(0);
			Tensor* _out = nullptr;
			TensorShape out_shape({num_boxes, 5});
			OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &_out));

			typename TTypes<float, 2>::Tensor out = _out->tensor<float, 2>();

			/* If a box the highest IoU with a gt out of all the boxes, it gets
			 * a positive label.  This code is here to keep track of which
			 * boxes have the highest IoU for a given ground truth.
			 * The first element of the pair is the box, and the second its IoU
			 */

			const int NEGATIVE = 0;
		   	const int NEITHER = -1;

			std::vector< std::pair<int, float> > bestIoU(num_gts);
			for(int i=0; i < num_gts; ++i) {
				bestIoU[i] = std::pair<int, float>(-1, 0.0);
			}

			for(int i=0; i < num_boxes; ++i) {
				float max_IoU = 0.;
				int label = NEITHER;
				/* We start upcoming loop through ground truths at j=1 instead of j=0
				 * since the 0th slot is saved for the background class, which isn't
				 * an actual class
				 */
				for(int j=1; j < num_gts; ++j) {
					float IoU = ComputeIOU(boxes, gt, i, j);
					#ifdef DEBUG_LOOP
						std::cout << "Box " << i << " and gt " << j;
						std::cout << " have IoU " << IoU << "\n";
					#endif
					if (IoU > bestIoU[j].second) {
						// Means this is the best fit for the j'th ground truth so far
						bestIoU[j].first = i;
						bestIoU[j].second = IoU;
					}
					if (IoU >= max_IoU) {
						// Best fit for the ith box so far
						label = j;
						max_IoU = IoU;
					}
				}

				if (max_IoU < iou_threshold_neg) {
			   		label = NEGATIVE;
				} else if (max_IoU < iou_threshold_pos) {
					label = NEITHER;
				}


				out(i, 0) = boxes(i, 0);
				out(i, 1) = boxes(i, 1);
				out(i, 2) = boxes(i, 2);
				out(i, 3) = boxes(i, 3);
				out(i, 4) = static_cast<float>(label);
			}

			/* If a box has not yet been assigned to a ground truth, then if it happens
			 * to have the best IoU with some ground truth out of all the boxes, assign
			 * that box to that ground truth
			 */
			for(int j=0; j < num_gts; ++j) {
				if (bestIoU[j].first >= 0 && out(bestIoU[j].first, 4) < 1) {
					out(bestIoU[j].first, 4) = static_cast<float>(j);
				}
			}

		}
	private:
		float iou_threshold_pos;
		float iou_threshold_neg;
};

REGISTER_KERNEL_BUILDER(\
		Name("IouLabeler")\
		.Device(DEVICE_CPU)\
		,IouLabelerOp);
