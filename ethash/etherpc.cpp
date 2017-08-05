#include "cpuminer-config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <unistd.h>
//#include <jansson.h>

#include <curl/curl.h>

#ifdef WIN32
#include <winsock2.h>
#include <mstcpip.h>
#else
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif

#include "miner.h"

struct data_buffer {
	void		*buf;
	size_t		len;
};

struct upload_buffer {
	const void	*buf;
	size_t		len;
	size_t		pos;
};

struct header_info {
	char		*lp_path;
	char		*reason;
	char		*stratum_url;
};

static void databuf_free(struct data_buffer *db)
{
	if (!db)
		return;

	free(db->buf);

	memset(db, 0, sizeof(*db));
}

static size_t all_data_cb(const void *ptr, size_t size, size_t nmemb,
			  void *user_data)
{
	struct data_buffer *db = (struct data_buffer *)user_data;
	size_t len = size * nmemb;
	size_t oldlen, newlen;
	void *newmem;
	static const uchar zero = 0;

	oldlen = db->len;
	newlen = oldlen + len;

	newmem = realloc(db->buf, newlen + 1);
	if (!newmem)
		return 0;

	db->buf = newmem;
	db->len = newlen;
	memcpy((char*)db->buf + oldlen, ptr, len);
	memcpy((char*)db->buf + newlen, &zero, 1);	/* null terminate */

	return len;
}

static size_t upload_data_cb(void *ptr, size_t size, size_t nmemb,
			     void *user_data)
{
	struct upload_buffer *ub = (struct upload_buffer *)user_data;
	unsigned int len = (unsigned int)(size * nmemb);

	if (len > ub->len - ub->pos)
		len = (unsigned int)(ub->len - ub->pos);

	if (len) {
		memcpy(ptr, (char*)ub->buf + ub->pos, len);
		ub->pos += len;
	}

	return len;
}

#if LIBCURL_VERSION_NUM >= 0x071200
static int seek_data_cb(void *user_data, curl_off_t offset, int origin)
{
	struct upload_buffer *ub = (struct upload_buffer *)user_data;

	switch (origin) {
	case SEEK_SET:
		ub->pos = (size_t)offset;
		break;
	case SEEK_CUR:
		ub->pos += (size_t)offset;
		break;
	case SEEK_END:
		ub->pos = ub->len + (size_t)offset;
		break;
	default:
		return 1; /* CURL_SEEKFUNC_FAIL */
	}

	return 0; /* CURL_SEEKFUNC_OK */
}
#endif

static size_t resp_hdr_cb(void *ptr, size_t size, size_t nmemb, void *user_data)
{
	struct header_info *hi = (struct header_info *)user_data;
	size_t remlen, slen, ptrlen = size * nmemb;
	char *rem, *val = NULL, *key = NULL;
	void *tmp;

	val = (char*)calloc(1, ptrlen);
	key = (char*)calloc(1, ptrlen);
	if (!key || !val)
		goto out;

	tmp = memchr(ptr, ':', ptrlen);
	if (!tmp || (tmp == ptr))	/* skip empty keys / blanks */
		goto out;
	slen = (size_t)((char*)tmp - (char*)ptr);
	if ((slen + 1) == ptrlen)	/* skip key w/ no value */
		goto out;
	memcpy(key, ptr, slen);		/* store & nul term key */
	key[slen] = 0;

	rem = (char*)ptr + slen + 1;		/* trim value's leading whitespace */
	remlen = ptrlen - slen - 1;
	while ((remlen > 0) && (isspace(*rem))) {
		remlen--;
		rem++;
	}

	memcpy(val, rem, remlen);	/* store value, trim trailing ws */
	val[remlen] = 0;
	while ((*val) && (isspace(val[strlen(val) - 1]))) {
		val[strlen(val) - 1] = 0;
	}
	if (!*val)			/* skip blank value */
		goto out;

	if (!strcasecmp("X-Long-Polling", key)) {
		hi->lp_path = val;	/* X-Mining-Extensions: longpoll */
		val = NULL;
	}

	if (!strcasecmp("X-Reject-Reason", key)) {
		hi->reason = val;	/* X-Mining-Extensions: reject-reason */
		//applog(LOG_WARNING, "%s:%s", key, val);
		val = NULL;
	}

	if (!strcasecmp("X-Stratum", key)) {
		hi->stratum_url = val;	/* steal memory reference */
		val = NULL;
	}

	if (!strcasecmp("X-Nonce-Range", key)) {
		/* todo when available: X-Mining-Extensions: noncerange */
	}
out:
	free(key);
	free(val);
	return ptrlen;
}

#if LIBCURL_VERSION_NUM >= 0x070f06
static int sockopt_keepalive_cb(void *userdata, curl_socket_t fd,
	curlsocktype purpose)
{
	int keepalive = 1;
	int tcp_keepcnt = 3;
	int tcp_keepidle = 50;
	int tcp_keepintvl = 50;
#ifdef WIN32
	DWORD outputBytes;
#endif

#ifndef WIN32
	if (unlikely(setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive,
		sizeof(keepalive))))
		return 1;
#ifdef __linux
	if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPCNT,
		&tcp_keepcnt, sizeof(tcp_keepcnt))))
		return 1;
	if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPIDLE,
		&tcp_keepidle, sizeof(tcp_keepidle))))
		return 1;
	if (unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPINTVL,
		&tcp_keepintvl, sizeof(tcp_keepintvl))))
		return 1;
#endif /* __linux */
#ifdef __APPLE_CC__
	if (unlikely(setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE,
		&tcp_keepintvl, sizeof(tcp_keepintvl))))
		return 1;
#endif /* __APPLE_CC__ */
#else /* WIN32 */
	struct tcp_keepalive vals;
	vals.onoff = 1;
	vals.keepalivetime = tcp_keepidle * 1000;
	vals.keepaliveinterval = tcp_keepintvl * 1000;
	if (unlikely(WSAIoctl(fd, SIO_KEEPALIVE_VALS, &vals, sizeof(vals),
		NULL, 0, &outputBytes, NULL, NULL)))
		return 1;
#endif /* WIN32 */

	return 0;
}
#endif

static json_t *ether_json_rpc(CURL *curl, const char *url, const char *rpc_req, int *curl_err)
{
	char len_hdr[64], hashrate_hdr[64];
	char curl_err_str[CURL_ERROR_SIZE] = { 0 };
	struct data_buffer all_data = { 0 };
	struct upload_buffer upload_data;
	struct curl_slist *headers = NULL;
	struct header_info hi = { 0 };
	char *httpdata;
	json_t *val, *err_val, *res_val;
	json_error_t err;
	int rc;

	long timeout = opt_timeout;
	bool keepalive = false;

	/* it is assumed that 'curl' is freshly [re]initialized at this pt */

	if (opt_protocol)
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	curl_easy_setopt(curl, CURLOPT_URL, url);
	if (opt_cert)
		curl_easy_setopt(curl, CURLOPT_CAINFO, opt_cert);
	curl_easy_setopt(curl, CURLOPT_ENCODING, "");
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
	curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, all_data_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &all_data);
	curl_easy_setopt(curl, CURLOPT_READFUNCTION, upload_data_cb);
	curl_easy_setopt(curl, CURLOPT_READDATA, &upload_data);
#if LIBCURL_VERSION_NUM >= 0x071200
	curl_easy_setopt(curl, CURLOPT_SEEKFUNCTION, &seek_data_cb);
	curl_easy_setopt(curl, CURLOPT_SEEKDATA, &upload_data);
#endif
	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_err_str);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
	curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, resp_hdr_cb);
	curl_easy_setopt(curl, CURLOPT_HEADERDATA, &hi);
	if (opt_proxy) {
		curl_easy_setopt(curl, CURLOPT_PROXY, opt_proxy);
		curl_easy_setopt(curl, CURLOPT_PROXYTYPE, opt_proxy_type);
	}
#if 0
	if (userpass) {
		curl_easy_setopt(curl, CURLOPT_USERPWD, userpass);
		curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
	}
#endif
#if LIBCURL_VERSION_NUM >= 0x070f06
	if (keepalive)
		curl_easy_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_keepalive_cb);
#endif
	curl_easy_setopt(curl, CURLOPT_POST, 1);

	if (opt_protocol)
		applog(LOG_DEBUG, "JSON protocol request:\n%s", rpc_req);

	upload_data.buf = rpc_req;
	upload_data.len = strlen(rpc_req);
	upload_data.pos = 0;
	sprintf(len_hdr, "Content-Length: %lu", (unsigned long) upload_data.len);
	sprintf(hashrate_hdr, "X-Mining-Hashrate: %llu", (unsigned long long) global_hashrate);

	headers = curl_slist_append(headers, "Content-Type: application/json");
	headers = curl_slist_append(headers, len_hdr);
	headers = curl_slist_append(headers, "User-Agent: " USER_AGENT);
	headers = curl_slist_append(headers, "X-Mining-Extensions: longpoll noncerange reject-reason");
	headers = curl_slist_append(headers, hashrate_hdr);
	headers = curl_slist_append(headers, "Accept:"); /* disable Accept hdr*/
	headers = curl_slist_append(headers, "Expect:"); /* disable Expect hdr*/

	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

	rc = curl_easy_perform(curl);
	if (curl_err != NULL)
		*curl_err = rc;
	if (rc) {
		if (rc != CURLE_OPERATION_TIMEDOUT) {
			applog(LOG_ERR, "HTTP request failed: %s", curl_err_str);
			goto err_out;
		}
	}

	if (!all_data.buf || !all_data.len) {
		applog(LOG_ERR, "Empty data received in json_rpc_call.");
		goto err_out;
	}

	httpdata = (char*) all_data.buf;

	if (*httpdata != '{' && *httpdata != '[') {
		long errcode = 0;
		CURLcode c = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &errcode);
		if (c == CURLE_OK && errcode == 401) {
			applog(LOG_ERR, "You are not authorized, check your login and password.");
			goto err_out;
		}
	}

	val = JSON_LOADS(httpdata, &err);
	if (!val) {
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		if (opt_protocol)
			applog(LOG_DEBUG, "%s", httpdata);
		goto err_out;
	}

	if (opt_protocol) {
		char *s = json_dumps(val, JSON_INDENT(3));
		applog(LOG_DEBUG, "JSON protocol response:\n%s\n", s);
		free(s);
	}

	/* JSON-RPC valid response returns a non-null 'result',
	 * and a null 'error'. */
	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if (!res_val || json_is_null(res_val) ||
	    (err_val && !json_is_null(err_val))) {
		char *s = NULL;

		if (err_val) {
			s = json_dumps(err_val, 0);
			json_t *msg = json_object_get(err_val, "message");
			json_t *err_code = json_object_get(err_val, "code");
			if (curl_err && json_integer_value(err_code))
				*curl_err = (int) json_integer_value(err_code);

			if (json_is_string(msg)) {
				free(s);
				s = strdup(json_string_value(msg));
				if (have_longpoll && s && !strcmp(s, "method not getwork")) {
					json_decref(err_val);
					free(s);
					goto err_out;
				}
			}
			json_decref(err_val);
		}
		else
			s = strdup("(unknown reason)");

		if (!curl_err || opt_debug)
			applog(LOG_ERR, "JSON-RPC call failed: %s", s);

		free(s);

		goto err_out;
	}

	if (hi.reason)
		json_object_set_new(val, "reject-reason", json_string(hi.reason));

	databuf_free(&all_data);
	curl_slist_free_all(headers);
	curl_easy_reset(curl);
	return val;

err_out:
	free(hi.lp_path);
	free(hi.reason);
	free(hi.stratum_url);
	databuf_free(&all_data);
	curl_slist_free_all(headers);
	curl_easy_reset(curl);
	return NULL;
}

// Made for len 32
bool hex2uint(uint32_t *target, const char *hexstr, int len)
{
	char buf[32];
	char pad[80];
	char hex_byte[3];
	char *ep;
	char *p = buf;
	char *hex = pad;

	hex_byte[2] = '\0';

	// pad, we can receive non padded numbers (0x44 => 0x0044)
	size_t hexlen = strlen(hexstr);
	if (hexlen < len*2)
		memset(hex, '0', len*2 - hexlen);
	sprintf(&hex[len*2 - hexlen],"%s", hexstr);

	while (*hex && len) {
		if (!hex[1]) {
			applog(LOG_ERR, "hex2uint str truncated");
			return false;
		}
		hex_byte[0] = hex[0];
		hex_byte[1] = hex[1];
		*p = (uchar) strtol(hex_byte, &ep, 16);
		if (*ep) {
			applog(LOG_ERR, "hex2uint failed on '%s'", hex_byte);
			return false;
		}
		p++;
		hex += 2;
		len--;
		if (len % 8 == 0) {
			//..
		}
	}

	for (int i = 0; i < 8; i++) {
		be32enc(&target[7 - i], ((uint32_t*)buf)[i]);
	}

	return (len == 0 && *hexstr == 0) ? true : false;
}

void uint2hex(char *out, uint32_t *in, int len)
{
	if (out) {
		for (int i = 0; i < len/sizeof(uint32_t); i++)
			sprintf(out + (i * 8), "%08x", in[7-i]);
	}
}

bool ether_getwork(CURL *curl, struct pool_infos *pool, struct work *work)
{
	char req[512];
	bool ret = false;

	sprintf(req, "{ \"jsonrpc\": \"2.0\", \"method\": \"eth_getWork\", \"params\": [], \"id\": 4 }\r\n");
	json_t *result = ether_json_rpc(curl, pool->url, req, NULL);
	if (result)
		ret = true;

	json_t *arr = json_object_get(result, "result");
	if (!json_is_array(arr)) {
		return false;
	}

	const char *header = json_string_value(json_array_get(arr, 0));
	const char *seed   = json_string_value(json_array_get(arr, 1));
	const char *target = json_string_value(json_array_get(arr, 2));

	//0x2b800e3ffcafdf7ffdd5b2ea1c0ed79c534e92b3459331eae61597943d0743bc
	//0x356e5a2cc1eba076e650ac7473fccc37952b46bc2e419a200cec0c451dce2336
	//0x000000000050ec7dea051867e987e3eed0066a128f161b5776a8f5da326b06ec

	applog(LOG_NOTICE, "%s %s %s", header, seed, target);

	hex2uint(work->data, &header[2], 32);
	hex2uint(&work->data[8], &seed[2], 32);
	hex2uint(work->target, &target[2], 32);

	//applog(LOG_BLUE, "%08x %08x", work->target[7], work->target[6]);

	return ret;
}

bool ether_submitwork(CURL *curl, struct pool_infos *pool, struct work *work)
{
	char req[512];
	bool ret = false;
	char noncehex[32]={0}, header[80]={0}, mixhash[80]={0};

	sprintf(noncehex, "0x%08x%08x", work->data[20], work->data[19]);
	uint2hex(header, work->data, 32);
	uint2hex(mixhash, &work->data[24], 32);

	applog(LOG_NOTICE, "%s mixhash:", __func__);
	applog_hash((uchar*)&work->data[24]);

	sprintf(req, "{ \"jsonrpc\": \"2.0\", \"method\": \"eth_submitWork\","
		"\"params\": [\"%s\", \"0x%s\", \"0x%s\"], \"id\": 4 }\r\n",
		noncehex, header, mixhash
	);
	json_t *result = ether_json_rpc(curl, pool->url, req, NULL);
	if (result)
		ret = true;

	json_t *res = json_object_get(result, "result");
	if (!json_is_true(res)) {
		applog(LOG_ERR, "not true! %s", json_dumps(result, JSON_INDENT(3)));
		exit(0);
		return false;
	}

	return ret;
}