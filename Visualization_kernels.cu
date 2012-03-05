/***************************************************************************/
/*                       CUDA based TLED Solver                            */
/*                     {c} 2008-2010 Karsten Noe                           */
/*                      The Alexandra Institute                            */
/*                   See our blog on cg.alexandra.dk                       */ 
/***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "TetrahedralMesh.h"
#include "cutil_math.h"
//#include "float_utils.h"

inline __host__ __device__ float3 crop_last_dim(float4 ui)
{
	return make_float3( ui.x, ui.y, ui.z );
}

__global__ void
extractSurface_k(float3 *tris, Tetrahedron *tetrahedra, Point *points, unsigned int numTets)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numTets)
		return;

	int zeroVertices[4];
	int numZeroVertices = 0;

	Tetrahedron tet = tetrahedra[me_idx];

	if (tet.x<0) return; // illegal tetrahedron

//	if (points[tet.x].distance==0)
			zeroVertices[numZeroVertices++] = tet.x;
//	if (points[tet.y].distance==0)
			zeroVertices[numZeroVertices++] = tet.y;
//	if (points[tet.z].distance==0)
			zeroVertices[numZeroVertices++] = tet.z;
//	if (points[tet.w].distance==0)
			zeroVertices[numZeroVertices++] = tet.w;

//	printf("numZeroes: %i", numZeroVertices);

	if (numZeroVertices>=3 )
	{
		for (int i=0; i<3; i++)
			tris[(3*me_idx)+i] = crop_last_dim(points[zeroVertices[i]]);
	}
	else
	{
		for (int i=0; i<3; i++)
			tris[(3*me_idx)+i] = make_float3(0,0,0);
	}
}


__global__ void
extractSurfaceWithDisplacements_k(float3 *tris, Tetrahedron *tetrahedra, Point *points, float4 *displacements, unsigned int numTets)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numTets)
		return;

	int4 tet = tetrahedra[me_idx];

	if (tet.x<0) return; // illegal tetrahedron

	int zeroVertices[4];
	int numZeroVertices = 0;


//	if (points[tetrahedra[me_idx].x].distance==0)
			zeroVertices[numZeroVertices++] = tet.x;
//	if (points[tetrahedra[me_idx].y].distance==0)
			zeroVertices[numZeroVertices++] = tet.y;
//	if (points[tetrahedra[me_idx].z].distance==0)
			zeroVertices[numZeroVertices++] = tet.z;
//	if (points[tetrahedra[me_idx].w].distance==0)
			zeroVertices[numZeroVertices++] = tet.w;

	//	printf("numZeroes: %i", numZeroVertices);

	if (numZeroVertices>=3)
	{
		for (int i=0; i<3; i++)
		{
			float3 pos = crop_last_dim(points[zeroVertices[i]]);
			float3 displacement = crop_last_dim(displacements[zeroVertices[i]]);
			pos.x += displacement.x;  
			pos.y += displacement.y;  
			pos.z += displacement.z;  
			tris[(3*me_idx)+i] = pos;
		}

	}
	else
	{
		for (int i=0; i<3; i++)
			tris[(3*me_idx)+i] = make_float3(0,0,0);
	}
}
/*
// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}
*/

__device__
float3 calcNormal(float4 *v0, float4 *v1, float4 *v2)
{
    float3 edge0 = crop_last_dim(*v1 - *v0);

    float3 edge1 = crop_last_dim(*v2 - *v0);

    // note - it's faster to perform normalization in vertex shader rather than here
    return cross(edge0, edge1);
}


__global__ void
updateSurfacePositionsFromDisplacements_k(float3 *tris, float3 *normals, TriangleSurface surface, Point *points, float4 *displacements)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=surface.numFaces) return;

	Triangle triangle = surface.faces[me_idx];

	float4 pos, pos2, pos3, displacement;

	pos = points[triangle.x-1];
	displacement = displacements[triangle.x-1];
	pos.x += displacement.x;  
	pos.y += displacement.y;  
	pos.z += displacement.z;  
	tris[(3*me_idx)+0] = crop_last_dim(pos);

	pos2 = points[triangle.y-1];
	displacement = displacements[triangle.y-1];
	pos2.x += displacement.x;  
	pos2.y += displacement.y;  
	pos2.z += displacement.z;  
	tris[(3*me_idx)+1] = crop_last_dim(pos2);

	pos3 = points[triangle.z-1];
	displacement = displacements[triangle.z-1];
	pos3.x += displacement.x;  
	pos3.y += displacement.y;  
	pos3.z += displacement.z;  
	tris[(3*me_idx)+2] = crop_last_dim(pos3);

	float3 normal = calcNormal(&pos,&pos2,&pos3);

	normals[(3*me_idx)+0] = normal;
	normals[(3*me_idx)+1] = normal;
	normals[(3*me_idx)+2] = normal;
}
