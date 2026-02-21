#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* prompt = "";
    if (argc > 1) {
        prompt = argv[1];
    }

    printf("[vspec-chat] prompt=\"%s\"\n", prompt);
    printf("[vspec-chat] loader/tokenizer/sampling in C is not implemented yet.\n");
    printf("[vspec-chat] use the python prototype in Vspec-chat/python for now.\n");
    return 0;
}
