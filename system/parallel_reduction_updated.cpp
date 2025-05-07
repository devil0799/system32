#include<iostream>
#include<omp.h>
#include<chrono>

using namespace std;



int min_sequential(int arr[], int n){
    int minVal=arr[0];
    for(int i=0;i<n;i++){
        if(arr[i]<minVal){
            minVal=arr[i];
        }
    }
    return minVal;
}

int max_sequential(int arr[], int n){
    int maxVal=arr[0];
    for(int i=0;i<n;i++){
        if(arr[i]>maxVal){
            maxVal=arr[i];
        }
    }
    return maxVal;
}

int sum_sequential(int arr[], int n){
    int sum=0;
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }
    return sum;
}

double avg_sequential(int arr[], int n){
    return (double)sum_sequential(arr,n)/n;
}


// parallel 

int min_parallel(int arr[], int n){
    int minVal=arr[0];
    #pragma omp parallel for reduction(min:minVal)
    for(int i=0;i<n;i++){
        if(arr[i]<minVal){
            minVal=arr[i];
        }
    }
    return minVal;
}

int max_parallel(int arr[], int n){
    int maxVal=arr[0];
    #pragma omp parallel for reduction(max:maxVal)
    for(int i=0;i<n;i++){
        if(arr[i]>maxVal){
            maxVal=arr[i];
        }
    }
    return maxVal;
}

int sum_parallel(int arr[], int n){
    int sum=0;
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }
    return sum;
}

double avg_parallel(int arr[], int n){
    return (double)sum_parallel(arr,n)/n;
}

int main(){

    int n=1000000;
    int* arr=new int[n];

    for(int i=0;i<n;i++){
        arr[i]=i+1;
    }

    auto start=chrono::high_resolution_clock::now();
    

    cout<<"Sequential Results: ";
    cout<<"\nMinimum Value using Sequential: "<<min_sequential(arr, n);
    cout<<"\nMaximum Value using Sequential: "<<max_sequential(arr, n);
    cout<<"\nSum  using Sequential: "<<sum_sequential(arr, n);
    cout<<"\nAverage Value using Sequential: "<<avg_sequential(arr, n);
    auto end=chrono::high_resolution_clock::now();

    chrono::duration<double> seq_duration=end-start;
    cout<<"\nTime for sequential execution: "<<seq_duration.count()<<" seconds";

    auto start2=chrono::high_resolution_clock::now();

    cout<<"\n\nParallel Results: ";
    cout<<"\nMinimum Value using Sequential: "<<min_parallel(arr, n);
    cout<<"\nMaximum Value using Sequential: "<<max_parallel(arr, n);
    cout<<"\nSum  using Sequential: "<<sum_parallel(arr, n);
    cout<<"\nAverage Value using Sequential: "<<avg_parallel(arr, n);
    auto end2=chrono::high_resolution_clock::now();

    chrono::duration<double> par_duration=end2-start2;
    cout<<"\nDuration for parallel execution: "<<par_duration.count()<<" seconds";

    cout<<"\n\nTotal speedup(sequential time/parallel time): "<<seq_duration.count()/par_duration.count();


    return 0;
}