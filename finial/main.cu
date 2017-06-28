#include <cstdio>
#include <cstdlib>
#include <thrust/count.h>
#include <cassert>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <time.h>
#include <sstream>
#define MAXFEATURE 10
#define MAXDATA 2000000
#define MAXLABEL 20
#define MAXDEPTH 15
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


__global__ void calImpurity (DataNode *x_train, int numLabel, int targetF,int left, int right) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	x_train[pos].impurity = right- left;

	//no cut boundary
	if (pos >= right || pos <= left) return;

	//equal to targetFeature belong to right side
	if (x_train[pos].feature[targetF] == x_train[pos-1].feature[targetF]) return;

	//left
	//compute each label appear times
	double left_imp = 0;
	{
		double Labelcount[MAXLABEL] = {};
		for (int i = left; i < pos; i++) 
			Labelcount[ x_train[i].label ]++;
		//compute impurity
		for (int i = 0; i < numLabel; i++) 
			left_imp += (Labelcount[i]/(pos-left))*(Labelcount[i]/(pos-left) );
		//mul weight
		left_imp = 1 - left_imp;
		if (left_imp < 0) printf("error:%d, pos = %d, left = %d, numLabel = %d\n",left_imp, pos, left, numLabel);
		left_imp *= (pos-left);
	}
	//right
	//compute each label appear times
	double right_imp = 0;
	{
		double Labelcount[MAXLABEL] = {};
		for (int i = pos; i < right; i++) 
			Labelcount[ x_train[i].label ]++;
		//compute impurity
		for (int i = 0; i < numLabel; i++) 
			right_imp += (Labelcount[i]/(right-pos)) * (Labelcount[i]/(right-pos)) ;
		//mul weight
		right_imp = 1-right_imp;
		right_imp *= (right-pos);
	}
	x_train[pos].impurity = right_imp + left_imp;
	return;
}

struct cmp_feature {
	int32_t i;
	cmp_feature(int32_t i): i(i) {}
	__host__ __device__ bool operator()(const DataNode &ldn, const DataNode &rdn) const 
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
	__host__ __device__ bool operator()(const DataNode &ldn, const DataNode &rdn) const 
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
void rec_buildTree(DataNode *x_train, ClassifyNode *parent, int numX, int numLabel, int numF, int left, int right, int depth ) {

		//printf("left = %d, right = %d targetF = %d\n",left, right, targetF);
		//stop control
		/*DataNode *min_label = thrust::min_element( thrust::device,
						x_train+left, x_train+right,
						cmp_label() );
		DataNode *max_label = thrust::max_element( thrust::device,
						x_train+left, x_train+right,
						cmp_label() );
		DataNode *cpu_min_label = (DataNode *)malloc( sizeof(DataNode) );
		cudaMemcpy( cpu_min_label, min_label, sizeof(DataNode), cudaMemcpyDeviceToHost );
		DataNode *cpu_max_label = (DataNode *)malloc( sizeof(DataNode) );
		cudaMemcpy( cpu_max_label, max_label, sizeof(DataNode), cudaMemcpyDeviceToHost );
		*/
	{
		if (right - left < 2) {
			int labelcount = 0, maxlabelcount = 0, reslabel;
			for (int label = 0; label < numLabel; label++) {
				labelcount = thrust::count_if(thrust::device, x_train+left, x_train+right, count_label(label) );
				if (labelcount > maxlabelcount) {
					maxlabelcount = labelcount;
					reslabel = label;
				}
			}
			parent->featureId = -1;
			parent->left = NULL;
			parent->right = NULL;
			parent->label = reslabel;
			printf("dist cut ,depth = %d, set %d as label, rate = %lf, dist = %d\n", depth, reslabel, maxlabelcount/(double)(right-left), right-left);
				return;
			}
	}
	{
		for(int label = 0; label < numLabel; label++) {
			int labelcount = thrust::count_if(thrust::device, x_train+left, x_train+right, count_label(label) );
			if(labelcount/(double)(right-left) > 0.95) {
				parent->featureId = -1;
				parent->left = NULL;
				parent->right = NULL;
				parent->label = label;
				printf("rate cut  depth = %d, set %d as label, rate = %lf, dist = %d\n", depth, label, labelcount/(double)(right-left), right-left);
				return;
			}
		}
	}
		
		//create leaf node
		/*
		if (cpu_min_label->label == cpu_max_label->label) {
			parent->featureId = -1;
			parent->left = NULL;
			parent->right = NULL;
			parent->label = cpu_min_label->label;
			return;
		}
		*/
		/*
		{	
			printf("d = %d\n", depth);
			if (depth > MAXDEPTH) {
				int labelcount = 0, maxlabelcount = 0, reslabel;
				for (int label = 0; label < numLabel; label++) {
					labelcount = thrust::count_if(thrust::device, x_train+left, x_train+right, count_label(label) );
					if (labelcount > maxlabelcount) {
						maxlabelcount = labelcount;
						reslabel = label;
					}
				}
				parent->featureId = -1;
				parent->left = NULL;
				parent->right = NULL;
				parent->label = reslabel;
				printf("depth cut, depth = %d, set %d as label, rate = %lf, dist = %d\n", depth, reslabel, maxlabelcount/(double)(right-left), right-left);
				return;
			}
		}
		*/
	double best_min_impurity = 2147483647;
	int best_feature;
	int best_threshold;
	unsigned int best_pos;
	for (int targetF = 0; targetF < numF; targetF++) {
		//sort by target feature
		thrust::sort(thrust::device, x_train + left, x_train + right, cmp_feature(targetF));

		//calculate impurity for all cut
		calImpurity<<< (numX)/1024 + 1, (1<<10) >>>(x_train, numLabel, targetF, left, right);

		//find min impurity cut
		DataNode *min_impurity = thrust::min_element( thrust::device, x_train+left, x_train+right, cmp_impurity() );
		DataNode *cpu_min_impurity = (DataNode *)malloc( sizeof(DataNode) );
		cudaMemcpy( cpu_min_impurity, min_impurity, sizeof(DataNode), cudaMemcpyDeviceToHost );
		
		if (cpu_min_impurity-> impurity < best_min_impurity) {
			best_min_impurity = cpu_min_impurity->impurity;
			best_feature = targetF;
			best_threshold = cpu_min_impurity->feature[targetF];
			best_pos = min_impurity-x_train;
		}
		free(cpu_min_impurity);
	}
	//set classify tree node

	parent->threshold = best_threshold;
	parent->featureId = best_feature;

	//find cut position
	unsigned int shreshold_pos = best_pos;
		
	//sort on best axis
	thrust::sort(thrust::device, x_train + left, x_train + right, cmp_feature(best_feature));
	//dfs create calssify tree
	ClassifyNode *left_child = (ClassifyNode *)malloc( sizeof(ClassifyNode) );
	parent->left = left_child;
	rec_buildTree(x_train, left_child, numX, numLabel, numF, left, shreshold_pos, depth+1);

	ClassifyNode *right_child = (ClassifyNode *)malloc( sizeof(ClassifyNode) );
	parent->right = right_child;
	rec_buildTree(x_train, right_child, numX, numLabel, numF, shreshold_pos, right, depth+1);

	//free
	//free(cpu_min_label);
	//free(cpu_max_label);
	//free(cpu_min_impurity);
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
void shuffle(DataNode *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          DataNode t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}
DataNode gl_x_train[MAXDATA];
int main() {
	int H, W;
	scanf("%d%d", &H, &W);
	printf("Size of training data = (%d, %d)\n",H, W);
	int numX = 1000;//, label;
	int numF = 10;
	int numLabel = 10;
	int numT = 1;
		/*
		int label;
		for(int i = 0; i < H; i++)
				for(int j = 0; j < W; j++) {
						gl_x_train[numX].feature[0] = i;
						gl_x_train[numX].feature[1] = j;
						scanf("%d", &label);
						gl_x_train[numX].label = label;
						numX++;
				}
		*/
		
	for (int i = 0; i < numX; i++) {
		static char line[1024];
		scanf("%s", line);
		for (int j = 0; line[j]; j++) {
			if (line[j] == ',')	
				line[j] = ' ';
		}
		std::stringstream sin(line);
		for ( int f = 0; f < numF; f++) 
			sin >> gl_x_train[i].feature[f];
		sin >> gl_x_train[i].label;
	}
		
#ifdef DDEBUG
	for(int i = 0; i < numX; i++) {
		for(int j = 0; j < numF; j++)
			printf("%f ", gl_x_train[i].feature[j]);
			printf("%d\n",gl_x_train[i].label);
	}
#endif
/*
		//copy data to gpu
		cudaError_t st;
		DataNode *gpu_x_train;
		st = cudaMalloc( (void**)&gpu_x_train, numX*sizeof(DataNode) );
		assert(st == cudaSuccess);
		st = cudaMemcpy( gpu_x_train, gl_x_train, numX*sizeof(DataNode), cudaMemcpyHostToDevice );
		assert(st == cudaSuccess);

		//buildtree
		ClassifyNode *classifyTree = (ClassifyNode *)malloc( sizeof(ClassifyNode) );
		long start = clock();
		rec_buildTree(gpu_x_train, classifyTree, numX, numLabel, numF, 0, 0, numX);
		printf("time = %lf\n",(double)(clock()-start)/CLOCKS_PER_SEC);
*/
		//buildforest
	float forest_rate = 1;
	DataNode *gpu_x_train;
	cudaMalloc( (void**)&gpu_x_train, forest_rate*numX*sizeof(DataNode) );
	ClassifyNode * forest[numT];
	srand(time(NULL));
	for (int i = 0; i < numT; i++) {
		forest[i] = (ClassifyNode *)malloc( sizeof(ClassifyNode) );
		shuffle(gl_x_train, numX);
		cudaMemcpy( gpu_x_train, gl_x_train, forest_rate*numX*sizeof(DataNode), cudaMemcpyHostToDevice );
		long start = clock();
		rec_buildTree(gpu_x_train, forest[i], (int)numX*forest_rate, numLabel, numF ,0 , numX*forest_rate, 0);
		printf("time = %lf\n",(double)(clock()-start)/CLOCKS_PER_SEC);
	}
		//predict
	int *res;
	for (int f = 0; f < numT; f++) {
		int incorrect = 0;
		res = predict(gl_x_train, numX, forest[f]);
		for (int i = 0; i < numX; i++) 
			if(res[i] != gl_x_train[i].label)
				incorrect++;
		printf("accuracy = %lf\n", (numX-incorrect)/(double)numX );
	}
		//free
		//free(classifyTree);
	free(res);
}
