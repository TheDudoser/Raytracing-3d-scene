#version 130

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform vec3 u_pos;
uniform float u_time;
uniform sampler2D u_sample;
uniform float u_sample_part;
uniform vec2 u_seed1;
uniform vec2 u_seed2;

const float MAX_DIST = 9999.0;
const int MAX_REF = 10;
vec3 light = normalize(vec3(-0.5, 0.75, -1.0));

uvec4 R_STATE;

uint TausStep(uint z, int S1, int S2, int S3, uint M)
{
	uint b = (((z << S1) ^ z) >> S2);
	return (((z & M) << S3) ^ b);
}

uint LCGStep(uint z, uint A, uint C)
{
	return (A * z + C);
}

vec2 hash22(vec2 p)
{
	p += u_seed1.x;
	vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yzx+33.33);
	return fract((p3.xx+p3.yz)*p3.zy);
}

float random()
{
	R_STATE.x = TausStep(R_STATE.x, 13, 19, 12, uint(4294967294));
	R_STATE.y = TausStep(R_STATE.y, 2, 25, 4, uint(4294967288));
	R_STATE.z = TausStep(R_STATE.z, 3, 11, 17, uint(4294967280));
	R_STATE.w = LCGStep(R_STATE.w, uint(1664525), uint(1013904223));
	return 2.3283064365387e-10 * float((R_STATE.x ^ R_STATE.y ^ R_STATE.z ^ R_STATE.w));
}

vec3 randomOnSphere() {
	vec3 rand = vec3(random(), random(), random());
	float theta = rand.x * 2.0 * 3.14159265;
	float v = rand.y;
	float phi = acos(2.0 * v - 1.0);
	float r = pow(rand.z, 1.0 / 3.0);
	float x = r * sin(phi) * cos(theta);
	float y = r * sin(phi) * sin(theta);
	float z = r * cos(phi);
	return vec3(x, y, z);
}

mat2 rot(float a) {
	float s = sin(a);
	float c = cos(a);
	return mat2(c, -s, s, c);
}

vec2 sphIntersect(in vec3 ro, in vec3 rd, float ra) {
	float b = dot(ro, rd);
	float c = dot(ro, ro) - ra * ra;
	float h = b * b - c;
	if(h < 0.0) return vec2(-1.0);
	h = sqrt(h);
	return vec2(-b - h, -b + h);
}

vec2 boxIntersection(in vec3 ro, in vec3 rd, in vec3 rad, out vec3 oN)  {
	vec3 m = 1.0 / rd;
	vec3 n = m * ro;
	vec3 k = abs(m) * rad;
	vec3 t1 = -n - k;
	vec3 t2 = -n + k;
	float tN = max(max(t1.x, t1.y), t1.z);
	float tF = min(min(t2.x, t2.y), t2.z);
	if(tN > tF || tF < 0.0) return vec2(-1.0);
	oN = -sign(rd) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
	return vec2(tN, tF);
}

float plaIntersect(in vec3 ro, in vec3 rd, in vec4 p) {
	return -(dot(ro, p.xyz) + p.w) / dot(rd, p.xyz);
}

vec3 getSky(vec3 rd) {
	vec3 col = vec3(0.3, 0.6, 1.0);
	vec3 sun = vec3(0.95, 0.9, 1.0);
	sun *= max(0.0, pow(dot(rd, light), 256.0));
	col *= max(0.0, dot(light, vec3(0.0, 0.0, -1.0)));
	return clamp(sun + col * 0.01, 0.0, 1.0);
}

float torIntersect( in vec3 ro, in vec3 rd, in vec2 tor )
{
	float po = 1.0;
	float Ra2 = tor.x*tor.x;
	float ra2 = tor.y*tor.y;
	float m = dot(ro,ro);
	float n = dot(ro,rd);
	float k = (m + Ra2 - ra2)/2.0;
	float k3 = n;
	float k2 = n*n - Ra2*dot(rd.xy,rd.xy) + k;
	float k1 = n*k - Ra2*dot(rd.xy,ro.xy);
	float k0 = k*k - Ra2*dot(ro.xy,ro.xy);

	if( abs(k3*(k3*k3-k2)+k1) < 0.01 )
	{
		po = -1.0;
		float tmp=k1; k1=k3; k3=tmp;
		k0 = 1.0/k0;
		k1 = k1*k0;
		k2 = k2*k0;
		k3 = k3*k0;
	}

	float c2 = k2*2.0 - 3.0*k3*k3;
	float c1 = k3*(k3*k3-k2)+k1;
	float c0 = k3*(k3*(c2+2.0*k2)-8.0*k1)+4.0*k0;
	c2 /= 3.0;
	c1 *= 2.0;
	c0 /= 3.0;
	float Q = c2*c2 + c0;
	float R = c2*c2*c2 - 3.0*c2*c0 + c1*c1;
	float h = R*R - Q*Q*Q;

	if( h>=0.0 )
	{
		h = sqrt(h);
		float v = sign(R+h)*pow(abs(R+h),1.0/3.0); // cube root
		float u = sign(R-h)*pow(abs(R-h),1.0/3.0); // cube root
		vec2 s = vec2( (v+u)+4.0*c2, (v-u)*sqrt(3.0));
		float y = sqrt(0.5*(length(s)+s.x));
		float x = 0.5*s.y/y;
		float r = 2.0*c1/(x*x+y*y);
		float t1 =  x - r - k3; t1 = (po<0.0)?2.0/t1:t1;
		float t2 = -x - r - k3; t2 = (po<0.0)?2.0/t2:t2;
		float t = 1e20;
		if( t1>0.0 ) t=t1;
		if( t2>0.0 ) t=min(t,t2);
		return t;
	}

	float sQ = sqrt(Q);
	float w = sQ*cos( acos(-R/(sQ*Q)) / 3.0 );
	float d2 = -(w+c2); if( d2<0.0 ) return -1.0;
	float d1 = sqrt(d2);
	float h1 = sqrt(w - 2.0*c2 + c1/d1);
	float h2 = sqrt(w - 2.0*c2 - c1/d1);
	float t1 = -d1 - h1 - k3; t1 = (po<0.0)?2.0/t1:t1;
	float t2 = -d1 + h1 - k3; t2 = (po<0.0)?2.0/t2:t2;
	float t3 =  d1 - h2 - k3; t3 = (po<0.0)?2.0/t3:t3;
	float t4 =  d1 + h2 - k3; t4 = (po<0.0)?2.0/t4:t4;
	float t = 1e20;
	if( t1>0.0 ) t=t1;
	if( t2>0.0 ) t=min(t,t2);
	if( t3>0.0 ) t=min(t,t3);
	if( t4>0.0 ) t=min(t,t4);
	return t;
}

vec3 torNormal( in vec3 pos, vec2 tor )
{
	return normalize( pos*(dot(pos,pos)-tor.y*tor.y - tor.x*tor.x*vec3(1.0,1.0,-1.0)));
}

float gouIntersect( in vec3 ro, in vec3 rd, in float ka, float kb )
{
	float po = 1.0;
	vec3 rd2 = rd*rd; vec3 rd3 = rd2*rd;
	vec3 ro2 = ro*ro; vec3 ro3 = ro2*ro;
	float k4 = dot(rd2,rd2);
	float k3 = dot(ro ,rd3);
	float k2 = dot(ro2,rd2) - kb/6.0;
	float k1 = dot(ro3,rd ) - kb*dot(rd,ro)/2.0;
	float k0 = dot(ro2,ro2) + ka - kb*dot(ro,ro);
	k3 /= k4;
	k2 /= k4;
	k1 /= k4;
	k0 /= k4;
	float c2 = k2 - k3*(k3);
	float c1 = k1 + k3*(2.0*k3*k3-3.0*k2);
	float c0 = k0 + k3*(k3*(c2+k2)*3.0-4.0*k1);

	if( abs(c1) < 0.1*abs(c2) )
	{
		po = -1.0;
		float tmp=k1; k1=k3; k3=tmp;
		k0 = 1.0/k0;
		k1 = k1*k0;
		k2 = k2*k0;
		k3 = k3*k0;
		c2 = k2 - k3*(k3);
		c1 = k1 + k3*(2.0*k3*k3-3.0*k2);
		c0 = k0 + k3*(k3*(c2+k2)*3.0-4.0*k1);
	}

	c0 /= 3.0;
	float Q = c2*c2 + c0;
	float R = c2*c2*c2 - 3.0*c0*c2 + c1*c1;
	float h = R*R - Q*Q*Q;

	if( h>0.0 ) // 2 intersections
	{
		h = sqrt(h);
		float s = sign(R+h)*pow(abs(R+h),1.0/3.0); // cube root
		float u = sign(R-h)*pow(abs(R-h),1.0/3.0); // cube root
		float x = s+u+4.0*c2;
		float y = s-u;
		float ks = x*x + y*y*3.0;
		float k = sqrt(ks);
		float t = -0.5*po*abs(y)*sqrt(6.0/(k+x)) - 2.0*c1*(k+x)/(ks+x*k) - k3;
		return (po<0.0)?1.0/t:t;
	}

	// 4 intersections
	float sQ = sqrt(Q);
	float w = sQ*cos(acos(-R/(sQ*Q))/3.0);
	float d2 = -w - c2;
	if( d2<0.0 ) return -1.0; //no intersection
	float d1 = sqrt(d2);
	float h1 = sqrt(w - 2.0*c2 + c1/d1);
	float h2 = sqrt(w - 2.0*c2 - c1/d1);
	float t1 = -d1 - h1 - k3; t1 = (po<0.0)?1.0/t1:t1;
	float t2 = -d1 + h1 - k3; t2 = (po<0.0)?1.0/t2:t2;
	float t3 =  d1 - h2 - k3; t3 = (po<0.0)?1.0/t3:t3;
	float t4 =  d1 + h2 - k3; t4 = (po<0.0)?1.0/t4:t4;
	float t = 1e20;
	if( t1>0.0 ) t=t1;
	if( t2>0.0 ) t=min(t,t2);
	if( t3>0.0 ) t=min(t,t3);
	if( t4>0.0 ) t=min(t,t4);
	return t;
}

vec3 gouNormal( in vec3 pos, float ka, float kb )
{
	return normalize( 4.0*pos*pos*pos - 2.0*pos*kb );
}

float sdDeathStar( in vec3 p2, in float ra, float rb, in float d )
{
	// sampling independent computations (only depend on shape)
	float a = (ra*ra - rb*rb + d*d)/(2.0*d);
	float b = sqrt(max(ra*ra-a*a,0.0));

	// sampling dependant computations
	vec2 p = vec2( p2.x, length(p2.yz) );
	if( p.x*b-p.y*a > d*max(b-p.y,0.0) )
		return length(p-vec2(a,b));
	else return max( (length(p)-ra), -(length(p-vec2(d,0))-rb));
}

//vec3 calcNormal( in vec3 pos )
//{
//	vec2 e = vec2(1.0,-1.0)*0.5773;
//	const float eps = 0.0005;
//	return normalize( e.xyy*map( pos + e.xyy*eps ) +
//	e.yyx*map( pos + e.yyx*eps ) +
//	e.yxy*map( pos + e.yxy*eps ) +
//	e.xxx*map( pos + e.xxx*eps ) );
//}

float sph4Intersect( in vec3 ro, in vec3 rd, in float ra )
{
	float r2 = ra*ra;
	vec3 d2 = rd*rd; vec3 d3 = d2*rd;
	vec3 o2 = ro*ro; vec3 o3 = o2*ro;
	float ka = 1.0/dot(d2,d2);
	float k3 = ka* dot(ro,d3);
	float k2 = ka* dot(o2,d2);
	float k1 = ka* dot(o3,rd);
	float k0 = ka*(dot(o2,o2) - r2*r2);
	float c2 = k2 - k3*k3;
	float c1 = k1 + 2.0*k3*k3*k3 - 3.0*k3*k2;
	float c0 = k0 - 3.0*k3*k3*k3*k3 + 6.0*k3*k3*k2 - 4.0*k3*k1;
	float p = c2*c2 + c0/3.0;
	float q = c2*c2*c2 - c2*c0 + c1*c1;
	float h = q*q - p*p*p;
	if( h<0.0 ) return -1.0; //no intersection
	float sh = sqrt(h);
	float s = sign(q+sh)*pow(abs(q+sh),1.0/3.0); // cuberoot
	float t = sign(q-sh)*pow(abs(q-sh),1.0/3.0); // cuberoot
	vec2  w = vec2( s+t,s-t );
	vec2  v = vec2( w.x+c2*4.0, w.y*sqrt(3.0) )*0.5;
	float r = length(v);
	return -abs(v.y)/sqrt(r+v.x) - c1/r - k3;
}

vec3 sph4Normal( in vec3 pos )
{
	return normalize( pos*pos*pos );
}

vec4 castRay(inout vec3 ro, inout vec3 rd) {
	vec4 col;
	vec2 minIt = vec2(MAX_DIST);
	vec2 it;
	vec3 n;
	mat2x4 spheres[6];
	spheres[0][0] = vec4(-1.0, 0.0, -0.01, 1.0);
	spheres[1][0] = vec4(0.0, 3.0, -0.01, 1.0);
	spheres[2][0] = vec4(1.0, -2.0, -0.01, 1.0);
	spheres[3][0] = vec4(3.5, -1.0, 0.5, 0.5);
	spheres[4][0] = vec4(5, 5, -1.0, 0.5);
	spheres[5][0] = vec4(-5.5, -0.5, -0.01, 1.0);

	spheres[0][1] = vec4(1.0, 1.0, 1.0, -0.5);
	spheres[1][1] = vec4(1.0, 1.0, 1.0, 0.5);
	spheres[2][1] = vec4(1.0, 0.0, 0.5, 1.0);
	spheres[3][1] = vec4(1.0, 1.0, 1.0, -2.0);
	spheres[4][1] = vec4(0.5, 1.0, 0.5, -2.0);
	spheres[5][1] = vec4(0.5, 0.5, 0.5, 0.0);
	for(int i = 0; i < spheres.length(); i++) {
		it = sphIntersect(ro - spheres[i][0].xyz, rd, spheres[i][0].w);
		if(it.x > 0.0 && it.x < minIt.x) {
			minIt = it;
			vec3 itPos = ro + rd * it.x;
			n = normalize(itPos - spheres[i][0].xyz);
			col = spheres[i][1];
		}
	}

	vec3 boxN;
	vec3 boxPos = vec3(3.0, 1.0, -0.001);
	it = boxIntersection(ro - boxPos, rd, vec3(1.0), boxN);
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = boxN;
		col = vec4(0.9, 0.2, 0.2, -0.5);
	}

	// импровизированный светильник
	vec3 box1;
	vec3 boxPos1 = vec3(10.0, 10.0, -0.001);
	it = boxIntersection(ro - boxPos1 + 15, rd, vec3(6.0, 6.0, 0.1), box1);
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = box1;
		col = vec4(1.0, 1.0, 1.0, -2.0);
	}

	// пол
	vec3 planeNormal = vec3(0.0, 0.0, -1.0);
	it = vec2(plaIntersect(ro, rd, vec4(planeNormal, 1.0)));
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = planeNormal;
		col = vec4(0.5, 0.25, 0.1, 0.0);
	}

	// стена 1
	vec3 planeNormal1 = vec3(0.09, 0, 0);
	it = vec2(plaIntersect(ro, rd, vec4(planeNormal1, 1.0)));
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = planeNormal1;
		col = vec4(0.9, 0.1, 0.1, 0.5);
	}

	// стена 2 (напротив стены 1)
	vec3 planeNormal2 = vec3(-0.09, 0, 0);
	it = vec2(plaIntersect(ro, rd, vec4(planeNormal2, 1.0)));
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = planeNormal2;
		col = vec4(0.1, 0.1, 0.9, 0.5);
	}

	// стена 3
	vec3 planeNormal3 = vec3(0.0, 0.09, 0);
	it = vec2(plaIntersect(ro, rd, vec4(planeNormal3, 1.0)));
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = planeNormal3;
		col = vec4(0.9, 0.9, 0.9, 0.5);
	}

	// стена 4 (напротив стены 3)
	vec3 planeNormal4 = vec3(0.0, -0.09, 0);
	it = vec2(plaIntersect(ro, rd, vec4(planeNormal4, 1.0)));
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = planeNormal4;
		col = vec4(0.9, 0.9, 0.9, 0.5);
	}

	// потолок
	vec3 planeNormal5 = vec3(0.0, 0.0, 0.067);
	it = vec2(plaIntersect(ro, rd, vec4(planeNormal5, 1.0)));
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = planeNormal5;
		col = vec4(0.5, 0.25, 0.1, 0.0);
	}

	// тор
	vec2 tor = vec2(1.0,0.4);

	vec3 torPos = vec3(-3.0, -3.0, 0.6);
	it = vec2(torIntersect(ro - torPos, rd, tor), 1);
	if(it.x > 0.0  && it.x < minIt.x) {
		minIt = it;
		n = torNormal(torPos, tor);
		//n = normalize(torPos*(dot(torPos,torPos)-tor.y*tor.y - tor.x*tor.x*vec3(1.0,1.0,-1.0)));
		col = vec4(0.4, 0.5, 0.6, 0.0);
	}

	// куб со скруглёнными гранями
	vec3 sph4 = vec3(-4.0, 4.0, 0.0);
	it = vec2(sph4Intersect(ro - sph4, rd, 1.0), 1);
	if(it.x > 0.0  && it.x < minIt.x) {
		minIt = it;
		n = sph4Normal(sph4);
		col = vec4(0.2, 0.2, 0.5, 0.0);
	}

	vec3 box3;
	vec3 boxPos3 = vec3(-6.0, 7.0, 0.5);
	it = boxIntersection(ro - boxPos3, rd, vec3(0.5), box3);
	if(it.x > 0.0 && it.x < minIt.x) {
		minIt = it;
		n = box3;
		col = vec4(1.0, 1.0, 1.0, -2.0);
	}

	// куб с вырезом внутри
	vec2 gou = vec2(0.5,0.2);

	vec3 gouPos = vec3(5, 5, -1);
	float ka = 6;
	float kb = 4;

	it = vec2(gouIntersect(ro - gouPos, rd, ka, kb), 1);
	if(it.x > 0.0  && it.x < minIt.x) {
		minIt = it;
		//n = normalize( 4.0*gouPos*gouPos*gouPos - 2.0*gouPos*kb );
		n = gouNormal(gouPos, ka, kb);
		col = vec4(1.0, 1.0, 1.0, 0);
	}


//		vec3 ds = vec3(10, 0, -1);
//		float f1 = 0.5;
//		float f2 = 0.35;
//		float f3 = 0.5;
//
//		it = vec2(sdDeathStar(ds, f1, f2, f3), 1);
//		if(it.x > 0.0  && it.x < minIt.x) {
//			//minIt = it;
////			n = normalize(vec3(10, 0, -1));
//			col = vec4(1.0, 1.0, 1.0, 0);
//		}

	if(minIt.x == MAX_DIST) return vec4(getSky(rd), -2.0);
	if(col.a == -2.0) return col;
	vec3 reflected = reflect(rd, n);
	if(col.a < 0.0) {
		float fresnel = 1.0 - abs(dot(-rd, n));
		if(random() - 0.1 < fresnel * fresnel) {
			rd = reflected;
			return col;
		}
		ro += rd * (minIt.y + 0.001);
		rd = refract(rd, n, 1.0 / (1.0 - col.a));
		return col;
	}
	vec3 itPos = ro + rd * it.x;
	vec3 pos = itPos;
	vec2 vector = vec2(2,1);
	vec3 r = randomOnSphere();
	vec3 diffuse = normalize(r * dot(r, n));

//	vec3 diffuse = normalize(pos*(dot(pos,pos)-tor.y*tor.y - tor.x*tor.x*vec3(1.0,1.0,-1.0)));

	ro += rd * (minIt.x - 0.001);
	rd = mix(diffuse, reflected, col.a);
	return col;
}

vec3 traceRay(vec3 ro, vec3 rd) {
	vec3 col = vec3(1.0);
	for(int i = 0; i < MAX_REF; i++)
	{
		vec4 refCol = castRay(ro, rd);
		col *= refCol.rgb;
		if(refCol.a == -2.0) return col;
	}
	return vec3(0.0);
}

void main() {
	vec2 uv = (gl_TexCoord[0].xy - 0.5) * u_resolution / u_resolution.y;
	vec2 uvRes = hash22(uv + 1.0) * u_resolution + u_resolution;
	R_STATE.x = uint(u_seed1.x + uvRes.x);
	R_STATE.y = uint(u_seed1.y + uvRes.x);
	R_STATE.z = uint(u_seed2.x + uvRes.y);
	R_STATE.w = uint(u_seed2.y + uvRes.y);
	vec3 rayOrigin = u_pos;
	vec3 rayDirection = normalize(vec3(1.0, uv));
	rayDirection.zx *= rot(-u_mouse.y);
	rayDirection.xy *= rot(u_mouse.x);
	vec3 col = vec3(0.0);
	int samples = 4;
	for(int i = 0; i < samples; i++) {
		col += traceRay(rayOrigin, rayDirection);
	}

	col /= samples;
	float white = 20.0;
	col *= white * 16.0;
	col = (col * (1.0 + col / white / white)) / (1.0 + col);
	vec3 sampleCol = texture(u_sample, gl_TexCoord[0].xy).rgb;
	col = mix(sampleCol, col, u_sample_part);
	gl_FragColor = vec4(col, 1.0);
}