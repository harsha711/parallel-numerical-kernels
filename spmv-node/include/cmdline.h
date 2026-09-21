
#pragma once
#include <stdlib.h>
#include <string.h>

static inline char * get_arg(int argc, char ** argv, const char * key)
{
    char * val = NULL;

    size_t keylen = strlen(key);

    for(int i = 1; i < argc; i++){
        char * token = argv[i];
        
        if(strncmp(token, "--", 2) != 0)
            continue;
        token += 2;

        if(strncmp(token, key, keylen) != 0)
            continue;
        token += keylen;

        val = argv[i];
    }
    
    return val;
}

static inline char * get_argval(int argc, char ** argv, const char * key)
{
    char * val = NULL;

    size_t keylen = strlen(key);

    for(int i = 1; i < argc; i++){
        char * token = argv[i];
        
        if(strncmp(token, "--", 2) != 0)
            continue;
        token += 2;

        if(strncmp(token, key, keylen) != 0)
            continue;
        token += keylen;

        if(strncmp(token, "=", 1) != 0)
            continue;
        token += 1;

        val = token;
    }
    
    return val;
}

/* Integer option value, or `fallback` when the option is absent. */
static inline int get_argval_int(int argc, char ** argv, const char * key, int fallback)
{
    char * v = get_argval(argc, argv, key);
    return v ? atoi(v) : fallback;
}
