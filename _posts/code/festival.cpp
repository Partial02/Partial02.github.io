#include <iostream>
using namespace std;

#define MAX_DAY 1000
int dp[MAX_DAY];
double min_cost = 100000;
double mean_cost;

int main(void)
{
    ios_base :: sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int C, N, L;
    cin >> C;

    for (int c = 0; c < C; c++) {
        cin >> N >> L;
        for (int n = 0; n < N; n++) cin >> dp[n];

        for (int i = 0; i < N; i++) {
            double sum = 0;
            for (int j = 0; j < N - i; j++) {
                sum += dp[i + j];
                if (j >= L - 1) {
                    mean_cost = sum / (j + 1);
                    min_cost = mean_cost < min_cost ? mean_cost : min_cost;
                }
            }
        }
        printf("%.9lf\n", min_cost);
        min_cost = 100000;
    }
    return 0;
}