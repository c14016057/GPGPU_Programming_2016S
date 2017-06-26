#include <cstdio>
#include <cstdlib>
#include <thrust/count.h>
#include <cassert>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#define MAXFEATURE 10
#define MAXDATA 2000000
#define MAXLABEL 5
#define MAXDEPTH 10
#define noDDEBUG



typedef struct {
	double feature[MAXFEATURE];
	int label;
	double impurity;
} DataNode;

typedef struct ClassifyNode{
	int featureId;
	double threshold;
	int label;
	struct ClassifyNode *left, *right;
} ClassifyNode;


__global__ void calImpurity(DataNode *x_train, 
						    int numLabel, 
						    int targetF,
						    int left, int right
						   ) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	x_train[pos].impurity = right- left;

	//no cut boundary
	if (pos >= right || pos <= left) return;

	//equal to targetFeature belong to right side
	if (x_train[pos].feature[targetF] == x_train[pos-1].feature[targetF]) return;

	//left
	//compute each label appear times
	double left_imp = 1;
	{
			double Labelcount[MAXLABEL] = {};
			for (int i = left; i < pos; i++) 
					Labelcount[ x_train[i].label ]++;
			//compute impurity
			for (int i = 0; i < numLabel; i++) 
					left_imp *= Labelcount[i]/(pos-left);
			//mul weight
			left_imp *= (pos-left);
	}
	//right
	//compute each label appear times
	double right_imp = 1;
	{
			double Labelcount[MAXLABEL] = {};
			for (int i = pos; i < right; i++) 
					Labelcount[ x_train[i].label ]++;
			//compute impurity
			for (int i = 0; i < numLabel; i++) 
					right_imp *= Labelcount[i]/(right-pos);
			//mul weight
			right_imp *= (right-pos);
	}
	x_train[pos].impurity = right_imp + left_imp;
	return;
}

struct cmp_feature {
		 int32_t i;
		cmp_feature(int32_t i): i(i) {}
		__host__ __device__ bool operator()(const DataNode &ldn,
						const DataNode &rdn) const 
		{
				return ldn.feature[i] < rdn.feature[i];
		}
};
struct cmp_impurity {
		__host__ __device__ bool operator()(const DataNode &ldn,
						const DataNode &rdn) const 
		{
				return ldn.impurity < rdn.impurity;
		}
};
struct cmp_label {
		__host__ __device__ bool operator()(const DataNode &ldn,
						const DataNode &rdn) const 
		{
				return ldn.label < rdn.label;
		}
};
struct count_label {
		int32_t i;
		count_label(int32_t i): i(i) {}
		__host__ __device__ bool operator()(const DataNode &dn) const 
		{
				return dn.label == i;
		}
};
void rec_buildTree(DataNode *x_train,
				ClassifyNode *parent,
				int numX, int numLabel, int numF, int targetF,
				int left, 
				int right
				) {
		//printf("left = %d, right = %d targetF = %d\n",left, right, targetF);
		//stop control
		DataNode *min_label = thrust::min_element( thrust::device,
						x_train+left, x_train+right,
						cmp_label() );
		DataNode *max_label = thrust::max_element( thrust::device,
						x_train+left, x_train+right,
						cmp_label() );
		DataNode *cpu_min_label = (DataNode *)malloc( sizeof(DataNode) );
		cudaMemcpy( cpu_min_label, min_label, sizeof(DataNode), cudaMemcpyDeviceToHost );
		DataNode *cpu_max_label = (DataNode *)malloc( sizeof(DataNode) );
		cudaMemcpy( cpu_max_label, max_label, sizeof(DataNode), cudaMemcpyDeviceToHost );

			for(int label = 0; label < numLabel; label++) {
				int labelcount = thrust::count(thrust::device, x_train+left, x_train+right, count_label(label) );
				if(labelcount/(double)(right-left) > 0.9) {
					parent->featureId = -1;
					parent->left = NULL;
					parent->right = NULL;
					parent->label = label;
					return;
				}
			}
		
		//create leaf node								
		if (cpu_min_label->label == cpu_max_label->label) {
			parent->featureId = -1;
			parent->left = NULL;
			parent->right = NULL;
			parent->label = cpu_min_label->label;
			return;
		}

		//sort by target feature
		thrust::sort(thrust::device, x_train + left, x_train + right, cmp_feature(targetF));

		//calculate impurity for all cut
		calImpurity<<< (numX)/1024 + 1, (1<<10) >>>(x_train, numLabel, targetF, left, right);

		//find min impurity cut
		DataNode *min_impurity = thrust::min_element( thrust::device,
						x_train+left, x_train+right,
						cmp_impurity() );
		DataNode *cpu_min_impurity = (DataNode *)malloc( sizeof(DataNode) );
		cudaMemcpy( cpu_min_impurity, min_impurity, sizeof(DataNode), cudaMemcpyDeviceToHost );
		
		//set classify tree node
		parent->threshold = cpu_min_impurity->feature[targetF];
		parent->featureId = targetF;

		//find cut position
		unsigned int shreshold_pos = min_impurity - x_train;
		//printf("cut %lf\n", parent->threshold);
		//dfs create calssify tree
		ClassifyNode *left_child = (ClassifyNode *)malloc( sizeof(ClassifyNode) );
		parent->left = left_child;
		rec_buildTree(x_train, left_child, numX, numLabel, numF, (targetF+1)%numF, left, shreshold_pos);

		ClassifyNode *right_child = (ClassifyNode *)malloc( sizeof(ClassifyNode) );
		parent->right = right_child;
		rec_buildTree(x_train, right_child, numX, numLabel, numF, (targetF+1)%numF, shreshold_pos, right);

		//free
		free(cpu_min_label);
		free(cpu_max_label);
		free(cpu_min_impurity);
}
int rec_predict(DataNode *target, ClassifyNode *clTree) {
	if(clTree->featureId <0) return clTree->label;
	if(target->feature[clTree->featureId] < clTree->threshold)
		return rec_predict(target, clTree->left);
	if(target->feature[clTree->featureId] >= clTree->threshold)
		return rec_predict(target, clTree->right);
	printf("something error\n");
	return -1;
}

int * predict(DataNode *x_train, int numX, ClassifyNode *clTree) {
	int *ret = (int *)malloc( numX*sizeof(int) );
	for (int i = 0; i < numX; i++)
		ret[i] = rec_predict( &x_train[i], clTree);
	return ret;
}
DataNode gl_x_train[MAXDATA];
int main() {
		int H, W;
		scanf("%d%d", &H, &W);
		printf("Size of training data = (%d, %d)\n",H, W);
		int numX = 0, label;
		int numF = 2;
		int numLabel = 2;
		for(int i = 0; i < H; i++)
				for(int j = 0; j < W; j++) {
						gl_x_train[numX].feature[0] = i;
						gl_x_train[numX].feature[1] = j;
						scanf("%d", &label);
						gl_x_train[numX].label = label;
						numX++;
				}
#ifdef DDEBUG
		for(int i = 0; i < numX; i++) {
				for(int j = 0; j < numF; j++)
						printf("%f ", gl_x_train[i].feature[j]);
				printf("%d\n",gl_x_train[i].label);
		}
#endif

		//copy data to gpu
		cudaError_t st;
		DataNode *gpu_x_train;
		st = cudaMalloc( (void**)&gpu_x_train, numX*sizeof(DataNode) );
		assert(st == cudaSuccess);
		st = cudaMemcpy( gpu_x_train, gl_x_train, numX*sizeof(DataNode), cudaMemcpyHostToDevice );
		assert(st == cudaSuccess);

		//buildtree
		ClassifyNode *classifyTree = (ClassifyNode *)malloc( sizeof(ClassifyNode) );
		rec_buildTree(gpu_x_train, classifyTree, numX, numLabel, numF, 0, 0, numX);

		//predict
		int *res;
		int incorrect = 0;
		res = predict(gl_x_train, numX, classifyTree);
		for (int i = 0; i < numX; i++) 
			if(res[i] != gl_x_train[i].label)
				incorrect++;
		printf("accuracy = %lf\n", (numX-incorrect)/(double)numX );

		//free
		free(classifyTree);
		free(res);
}
