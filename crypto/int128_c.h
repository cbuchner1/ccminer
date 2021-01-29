#ifndef UINT128_C_H
#define UINT128_C_H

struct __uint128 {
	uint64_t Hi;
	uint64_t Lo;
};
typedef struct __uint128 uint128;

void Increment(uint128 * N)
{
	uint64_t T = (N->Lo + 1);
	N->Hi += ((N->Lo ^T) & N->Lo) >> 63;
	N->Lo = T;
}

void Decrement(uint128 * N)
{
	uint64_t T = (N->Lo - 1);
	N->Hi -= ((T ^ N->Lo) & T) >> 63;
	N->Lo = T;
}

void Add(uint128 * Ans, uint128 N, uint128 M)
{
	uint64_t C = (((N.Lo & M.Lo) & 1) + (N.Lo >> 1) + (M.Lo >> 1)) >> 63;
	Ans->Hi = N.Hi + M.Hi + C;
	Ans->Lo = N.Lo + M.Lo;
}

void Subtract(uint128 * Ans, uint128 N, uint128 M)
{
	Ans->Lo = N.Lo - M.Lo;
	uint64_t C = (((Ans->Lo & M.Lo) & 1) + (M.Lo >> 1) + (Ans->Lo >> 1)) >> 63;
	Ans->Hi = N.Hi - (M.Hi + C);
}

void inc128(uint128 N, uint128* A)
{
	A->Lo = (N.Lo + 1);
	A->Hi = N.Hi + (((N.Lo ^ A->Lo) & N.Lo) >> 63);
}

void dec128(uint128 N, uint128* A)
{
	A->Lo = N.Lo - 1;
	A->Hi = N.Hi - (((A->Lo ^ N.Lo) & A->Lo) >> 63);
}

void add128(uint128 N, uint128 M, uint128* A)
{
	uint64_t C = (((N.Lo & M.Lo) & 1) + (N.Lo >> 1) + (M.Lo >> 1)) >> 63;
	A->Hi = N.Hi + M.Hi + C;
	A->Lo = N.Lo + M.Lo;
}

void sub128(uint128 N, uint128 M, uint128* A)
{
	A->Lo = N.Lo - M.Lo;
	uint64_t C = (((A->Lo & M.Lo) & 1) + (M.Lo >> 1) + (A->Lo >> 1)) >> 63;
	A->Hi = N.Hi - (M.Hi + C);
}

void mult64to128(uint64_t u, uint64_t v, uint64_t * h, uint64_t *l)
{
	uint64_t u1 = (u & 0xffffffff);
	uint64_t v1 = (v & 0xffffffff);
	uint64_t t = (u1 * v1);
	uint64_t w3 = (t & 0xffffffff);
	uint64_t k = (t >> 32);

	u >>= 32;
	t = (u * v1) + k;
	k = (t & 0xffffffff);
	uint64_t w1 = (t >> 32);

	v >>= 32;
	t = (u1 * v) + k;
	k = (t >> 32);

	*h = (u * v) + w1 + k;
	*l = (t << 32) + w3;
}

void mult128(uint128 N, uint128 M, uint128 * Ans)
{
	mult64to128(N.Lo, M.Lo, &Ans->Hi, &Ans->Lo);
	Ans->Hi += (N.Hi * M.Lo) + (N.Lo * M.Hi);
}

void mult128to256(uint128 N, uint128 M, uint128 * H, uint128 * L)
{
	mult64to128(N.Hi, M.Hi, &H->Hi, &H->Lo);
	mult64to128(N.Lo, M.Lo, &L->Hi, &L->Lo);

	uint128 T;
	mult64to128(N.Hi, M.Lo, &T.Hi, &T.Lo);
	L->Hi += T.Lo;
	if(L->Hi < T.Lo)  // if L->Hi overflowed
	{
		Increment(H);
	}
	H->Lo += T.Hi;
	if(H->Lo < T.Hi)  // if H->Lo overflowed
	{
		++H->Hi;
	}

	mult64to128(N.Lo, M.Hi, &T.Hi, &T.Lo);
	L->Hi += T.Lo;
	if(L->Hi < T.Lo)  // if L->Hi overflowed
	{
		Increment(H);
	}
	H->Lo += T.Hi;
	if(H->Lo < T.Hi)  // if H->Lo overflowed
	{
		++H->Hi;
	}
}


void sqr64to128(uint64_t r, uint64_t * h, uint64_t *l)
{
	uint64_t r1 = (r & 0xffffffff);
	uint64_t t = (r1 * r1);
	uint64_t w3 = (t & 0xffffffff);
	uint64_t k = (t >> 32);

	r >>= 32;
	uint64_t m = (r * r1);
	t = m + k;
	uint64_t w2 = (t & 0xffffffff);
	uint64_t w1 = (t >> 32);

	t = m + w2;
	k = (t >> 32);
	*h = (r * r) + w1 + k;
	*l = (t << 32) + w3;
}

void sqr128(uint128 R, uint128 * Ans)
{
	sqr64to128(R.Lo, &Ans->Hi, &Ans->Lo);
	Ans->Hi += (R.Hi * R.Lo) << 1;
}

void sqr128to256(uint128 R, uint128 * H, uint128 * L)
{
	sqr64to128(R.Hi, &H->Hi, &H->Lo);
	sqr64to128(R.Lo, &L->Hi, &L->Lo);

	uint128 T;
	mult64to128(R.Hi, R.Lo, &T.Hi, &T.Lo);

	H->Hi += (T.Hi >> 63);
	T.Hi = (T.Hi << 1) | (T.Lo >> 63);  // Shift Left 1 bit
	T.Lo <<= 1;

	L->Hi += T.Lo;
	if(L->Hi < T.Lo)  // if L->Hi overflowed
	{
		Increment(H);
	}

	H->Lo += T.Hi;
	if(H->Lo < T.Hi)  // if H->Lo overflowed
	{
		++H->Hi;
	}
}

void shiftleft128(uint128 N, size_t S, uint128 * A)
{
	uint64_t M1, M2;
	S &= 127;

	M1 = ((((S + 127) | S) & 64) >> 6) - 1llu;
	M2 = (S >> 6) - 1llu;
	S &= 63;
	A->Hi = (N.Lo << S) & (~M2);
	A->Lo = (N.Lo << S) & M2;
	A->Hi |= ((N.Hi << S) | ((N.Lo >> (64 - S)) & M1)) & M2;

/*
	S &= 127;

	if(S != 0)
	{
		if(S > 64)
		{
			A.Hi = N.Lo << (S - 64);
			A.Lo = 0;
		}
		else if(S < 64)
		{
			A.Hi = (N.Hi << S) | (N.Lo >> (64 - S));
			A.Lo = N.Lo << S;
		}
		else
		{
			A.Hi = N.Lo;
			A.Lo = 0;
		}
	}
	else
	{
		A.Hi = N.Hi;
		A.Lo = N.Lo;
	}
	//*/
}

void shiftright128(uint128 N, size_t S, uint128 * A)
{
	uint64_t M1, M2;
	S &= 127;

	M1 = ((((S + 127) | S) & 64) >> 6) - 1llu;
	M2 = (S >> 6) - 1llu;
	S &= 63;
	A->Lo = (N.Hi >> S) & (~M2);
	A->Hi = (N.Hi >> S) & M2;
	A->Lo |= ((N.Lo >> S) | ((N.Hi << (64 - S)) & M1)) & M2;

	/*
	S &= 127;

	if(S != 0)
	{
		if(S > 64)
		{
			A.Hi = N.Hi >> (S - 64);
			A.Lo = 0;
		}
		else if(S < 64)
		{
			A.Lo = (N.Lo >> S) | (N.Hi << (64 - S));
			A.Hi = N.Hi >> S;
		}
		else
		{
			A.Lo = N.Hi;
			A.Hi = 0;
		}
	}
	else
	{
		A.Hi = N.Hi;
		A.Lo = N.Lo;
	}
	//*/
}


void not128(uint128 N, uint128 * A)
{
	A->Hi = ~N.Hi;
	A->Lo = ~N.Lo;
}

void or128(uint128 N1, uint128 N2, uint128 * A)
{
	A->Hi = N1.Hi | N2.Hi;
	A->Lo = N1.Lo | N2.Lo;
}

void and128(uint128 N1, uint128 N2, uint128 * A)
{
	A->Hi = N1.Hi & N2.Hi;
	A->Lo = N1.Lo & N2.Lo;
}

void xor128(uint128 N1, uint128 N2, uint128 * A)
{
	A->Hi = N1.Hi ^ N2.Hi;
	A->Lo = N1.Lo ^ N2.Lo;
}

size_t nlz64(uint64_t N)
{
	uint64_t I;
	size_t C;

	I = ~N;
	C = ((I ^ (I + 1)) & I) >> 63;

	I = (N >> 32) + 0xffffffff;
	I = ((I & 0x100000000) ^ 0x100000000) >> 27;
	C += I;  N <<= I;

	I = (N >> 48) + 0xffff;
	I = ((I & 0x10000) ^ 0x10000) >> 12;
	C += I;  N <<= I;

	I = (N >> 56) + 0xff;
	I = ((I & 0x100) ^ 0x100) >> 5;
	C += I;  N <<= I;

	I = (N >> 60) + 0xf;
	I = ((I & 0x10) ^ 0x10) >> 2;
	C += I;  N <<= I;

	I = (N >> 62) + 3;
	I = ((I & 4) ^ 4) >> 1;
	C += I;  N <<= I;

	C += (N >> 63) ^ 1;

	return C;
}

size_t ntz64(uint64_t N)
{
	uint64_t I = ~N;
	size_t C = ((I ^ (I + 1)) & I) >> 63;

	I = (N & 0xffffffff) + 0xffffffff;
	I = ((I & 0x100000000) ^ 0x100000000) >> 27;
	C += I;  N >>= I;

	I = (N & 0xffff) + 0xffff;
	I = ((I & 0x10000) ^ 0x10000) >> 12;
	C += I;  N >>= I;

	I = (N & 0xff) + 0xff;
	I = ((I & 0x100) ^ 0x100) >> 5;
	C += I;  N >>= I;

	I = (N & 0xf) + 0xf;
	I = ((I & 0x10) ^ 0x10) >> 2;
	C += I;  N >>= I;

	I = (N & 3) + 3;
	I = ((I & 4) ^ 4) >> 1;
	C += I;  N >>= I;

	C += ((N & 1) ^ 1);

	return C;
}

size_t popcnt64(uint64_t V)
{
	// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
	V -= ((V >> 1) & 0x5555555555555555);
	V = (V & 0x3333333333333333) + ((V >> 2) & 0x3333333333333333);
	return ((V + (V >> 4) & 0xF0F0F0F0F0F0F0F) * 0x101010101010101) >> 56;
}

size_t popcnt128(uint128 N)
{
	return popcnt64(N.Hi) + popcnt64(N.Lo);
}


size_t nlz128(uint128 N)
{
	return (N.Hi == 0) ? nlz64(N.Lo) + 64 : nlz64(N.Hi);
}

size_t ntz128(uint128 N)
{
	return (N.Lo == 0) ? ntz64(N.Hi) + 64 : ntz64(N.Lo);
}
int compare128(uint128 N1, uint128 N2)
{
	return	(((N1.Hi > N2.Hi) || ((N1.Hi == N2.Hi) && (N1.Lo > N2.Lo))) ? 1 : 0)
		 -  (((N1.Hi < N2.Hi) || ((N1.Hi == N2.Hi) && (N1.Lo < N2.Lo))) ? 1 : 0);
}

void bindivmod128(uint128 M, uint128 N, uint128 * Q, uint128 *R)
{
	Q->Hi = Q->Lo = 0;
	size_t Shift = nlz128(N) - nlz128(M);
	shiftleft128(N, Shift, &N);

	do
	{
		shiftleft128(*Q, (size_t)1, Q);
		if(compare128(M, N) >= 0)
		{
			sub128(M, N, &M);
			Q->Lo |= 1;
		}

		shiftright128(N, 1, &N);
	}while(Shift-- != 0);

	R->Hi = M.Hi;
	R->Lo = M.Lo;
}

void divmod128by64(const uint64_t u1, const uint64_t u0, uint64_t v, uint64_t * q, uint64_t * r)
{
	const uint64_t b = 1ll << 32;
	uint64_t un1, un0, vn1, vn0, q1, q0, un32, un21, un10, rhat, left, right;
	size_t s;

	s = nlz64(v);
	v <<= s;
	vn1 = v >> 32;
	vn0 = v & 0xffffffff;

	if (s > 0)
	{
		un32 = (u1 << s) | (u0 >> (64 - s));
		un10 = u0 << s;
	}
	else
	{
		un32 = u1;
		un10 = u0;
	}

	un1 = un10 >> 32;
	un0 = un10 & 0xffffffff;

	q1 = un32 / vn1;
	rhat = un32 % vn1;

	left = q1 * vn0;
	right = (rhat << 32) + un1;
again1:
	if ((q1 >= b) || (left > right))
	{
		--q1;
		rhat += vn1;
		if (rhat < b)
		{
			left -= vn0;
			right = (rhat << 32) | un1;
			goto again1;
		}
	}

	un21 = (un32 << 32) + (un1 - (q1 * v));

	q0 = un21 / vn1;
	rhat = un21 % vn1;

	left = q0 * vn0;
	right = (rhat << 32) | un0;
again2:
	if ((q0 >= b) || (left > right))
	{
		--q0;
		rhat += vn1;
		if (rhat < b)
		{
			left -= vn0;
			right = (rhat << 32) | un0;
			goto again2;
		}
	}

	*r = ((un21 << 32) + (un0 - (q0 * v))) >> s;
	*q = (q1 << 32) | q0;
}

static void divmod128by128(uint128 M, uint128 N, uint128 * Q, uint128 * R)
{
	if (N.Hi == 0)
	{
		if (M.Hi < N.Lo)
		{
			divmod128by64(M.Hi, M.Lo, N.Lo, &Q->Lo, &R->Lo);
			Q->Hi = 0;
			R->Hi = 0;
			return;
		}
		else
		{
			Q->Hi = M.Hi / N.Lo;
			R->Hi = M.Hi % N.Lo;
			divmod128by64(R->Hi, M.Lo, N.Lo, &Q->Lo, &R->Lo);
			R->Hi = 0;
			return;
		}
	}
	else
	{
		size_t n = nlz64(N.Hi);

		uint128 v1;
		shiftleft128(N, n, &v1);

		uint128 u1;
		shiftright128(M, 1, &u1);

		uint128 q1;
		divmod128by64(u1.Hi, u1.Lo, v1.Hi, &q1.Hi, &q1.Lo);
		q1.Hi = 0;
		shiftright128(q1, 63 - n, &q1);

		if ((q1.Hi | q1.Lo) != 0)
		{
			dec128(q1, &q1);
		}

		Q->Hi = q1.Hi;
		Q->Lo = q1.Lo;
		mult128(q1, N, &q1);
		sub128(M, q1, R);

		if (compare128(*R, N) >= 0)
		{
			inc128(*Q, Q);
			sub128(*R, N, R);
		}

		return;
	}
}

void divmod128(uint128 M, uint128 N, uint128 * Q, uint128 * R)
{
	size_t Nlz, Mlz, Ntz;
	int C;

	Nlz = nlz128(N);
	Mlz = nlz128(M);
	Ntz = ntz128(N);

	if(Nlz == 128)
	{
		return;
	}
	else if((M.Hi | N.Hi) == 0)
	{
		Q->Hi = R->Hi = 0;
		Q->Lo = M.Lo / N.Lo;
		R->Lo = M.Lo % N.Lo;
		return;
	}
	else if(Nlz == 127)
	{
		*Q = M;
		R->Hi = R->Lo = 0;
		return;
	}
	else if((Ntz + Nlz) == 127)
	{
		shiftright128(M, Ntz, Q);
		dec128(N, &N);
		and128(N, M, R);
		return;
	}

	C = compare128(M, N);
	if(C < 0)
	{
		Q->Hi = Q->Lo = 0;
		*R = M;
		return;
	}
	else if(C == 0)
	{
		Q->Hi = R->Hi = R->Lo = 0;
		Q->Lo = 1;
		return;
	}

	if((Nlz - Mlz) > 5)
	{
		divmod128by128(M, N, Q, R);
	}
	else
	{
		bindivmod128(M, N, Q, R);
	}
}
#endif