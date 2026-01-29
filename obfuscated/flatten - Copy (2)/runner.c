#include <stdio.h>

// declare the obfuscated function (example)
int binarySearch(int arr[], int l, int r, int x);

int main() {
    int arr[] = {2, 3, 4, 10, 40};
    int n = sizeof(arr) / sizeof(arr[0]);
    int x = 10;

    int result = binarySearch(arr, 0, n - 1, x);

    if (result == -1)
        printf("Element not present\n");
    else
        printf("Element found at index %d\n", result);

    return 0;
}
