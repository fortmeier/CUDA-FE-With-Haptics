#ifndef Vector3DD_H
#define Vector3DD_H

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <vector_types.h>

using namespace std;

class Vector3D  {
public:
    
    Vector3D() { e[0] = 0; e[1] = 0; e[2] = 0;}
    Vector3D(float e0, float e1, float e2) {e[0]=e0; e[1]=e1; e[2]=e2;}
    float x() const { return e[0]; }
    float y() const { return e[1]; }
    float z() const { return e[2]; }
    void setX(float a) { e[0] = a; }
    void setY(float a) { e[1] = a; }
    void setZ(float a) { e[2] = a; }

    inline Vector3D(const Vector3D &v) {
         e[0] = v.e[0]; e[1] = v.e[1]; e[2] = v.e[2];
    }

    inline Vector3D(const float3 &v) {
         e[0] = v.x; e[1] = v.y; e[2] = v.z;
    }

	inline __device__ float3 make_float3() {
		float3 v;
         v.x = e[0];
         v.y = e[1];
         v.z = e[2];
		 return v;
    }

    const Vector3D& operator+() const { return *this; }
    Vector3D operator-() const { return Vector3D(-e[0], -e[1], -e[2]); }
    float& operator[](int i) {  return e[i]; }
    float operator[](int i) const { return e[i];}

    Vector3D& operator+=(const Vector3D &v2);
    Vector3D& operator-=(const Vector3D &v2);
    Vector3D& operator*=(const float t);
	Vector3D& operator*=(const Vector3D &v2);
    Vector3D& operator/=(const float t);
	Vector3D& operator/=(const Vector3D &v2);


    
    float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    float squaredLength() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }

    void makeUnitVector();

    
    float minComponent() const { return e[indexOfMinComponent()]; }
    float maxComponent() const { return e[indexOfMaxComponent()]; }
    float maxAbsComponent() const { return e[indexOfMaxAbsComponent()]; }
    float minAbsComponent() const { return e[indexOfMinAbsComponent()]; }
    int indexOfMinComponent() const { 
	return (e[0]< e[1] && e[0]< e[2]) ? 0 : (e[1] < e[2] ? 1 : 2);
    }

    int indexOfMinAbsComponent() const {
	if (fabs(e[0]) < fabs(e[1]) && fabs(e[0]) < fabs(e[2]))
	    return 0;
	else if (fabs(e[1]) < fabs(e[2]))
	    return 1;
	else
	    return 2;
    }

    int indexOfMaxComponent() const {
	return (e[0]> e[1] && e[0]> e[2]) ? 0 : (e[1] > e[2] ? 1 : 2);
    }

    int indexOfMaxAbsComponent() const {
	if (fabs(e[0]) > fabs(e[1]) && fabs(e[0]) > fabs(e[2]))
	    return 0;
	else if (fabs(e[1]) > fabs(e[2]))
	    return 1;
	else
	    return 2;
    }

    float e[3];
};


inline bool operator==(const Vector3D &t1, const Vector3D &t2) {
   return ((t1[0]==t2[0])&&(t1[1]==t2[1])&&(t1[2]==t2[2]));
}

inline bool operator!=(const Vector3D &t1, const Vector3D &t2) {
   return ((t1[0]!=t2[0])||(t1[1]!=t2[1])||(t1[2]!=t2[2]));
}

inline Vector3D unitVector(const Vector3D& v) {
    float k = 1.0f / sqrt(v.e[0]*v.e[0] + v.e[1]*v.e[1] + v.e[2]*v.e[2]);
    return Vector3D(v.e[0]*k, v.e[1]*k, v.e[2]*k);
}

inline void Vector3D::makeUnitVector() {
    float k = 1.0f / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

inline Vector3D operator+(const Vector3D &v1, const Vector3D &v2) {
    return Vector3D( v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

inline Vector3D operator-(const Vector3D &v1, const Vector3D &v2) {
    return Vector3D( v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

inline Vector3D operator*(float t, const Vector3D &v) {
    return Vector3D(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline Vector3D operator*(const Vector3D &v, float t) {
    return Vector3D(t*v.e[0], t*v.e[1], t*v.e[2]); 
}

inline Vector3D operator/(const Vector3D &v, float t) {
    return Vector3D(v.e[0]/t, v.e[1]/t, v.e[2]/t); 
}

inline Vector3D operator/(const Vector3D &v1, const Vector3D &v2) {
    return Vector3D(v1.e[0]/v2.e[0], v1.e[1]/v2.e[1], v1.e[2]/v2.e[2]); 
}

inline Vector3D operator*(const Vector3D &v1, const Vector3D &v2) {
    return Vector3D(v1.e[0]*v2.e[0], v1.e[1]*v2.e[1], v1.e[2]*v2.e[2]); 
}

inline bool operator<(const Vector3D &v1, const Vector3D &v2) {
	return v1.squaredLength()<v2.squaredLength();
}


inline float dot(const Vector3D &v1, const Vector3D &v2) {
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

inline Vector3D cross(const Vector3D &v1, const Vector3D &v2) {
    return Vector3D( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                      (v1.e[2]*v2.e[0] - v1.e[0]*v2.e[2]),
                      (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}


inline Vector3D& Vector3D::operator+=(const Vector3D &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

inline Vector3D& Vector3D::operator-=(const Vector3D& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

inline Vector3D& Vector3D::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

inline Vector3D& Vector3D::operator/=(const float t) {
    e[0]  /= t;
    e[1]  /= t;
    e[2]  /= t;
    return *this;
}

inline Vector3D& Vector3D::operator/=(const Vector3D &v) {
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

inline Vector3D& Vector3D::operator*=(const Vector3D &v) {
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

inline
Vector3D reflect(const Vector3D& in, const Vector3D& normal)
{
  // assumes unit length normal
  return in - normal * (2 * dot(in, normal));
}

#endif
