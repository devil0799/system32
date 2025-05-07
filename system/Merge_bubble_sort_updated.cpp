#include<iostream>
#include<vector>
#include<bits/stdc++.h>
#include<omp.h>
#include<chrono>

using namespace std;



void seq_bubble_sort(vector<int> &arr, int n){

    for(int i=0;i<n-1;i++){
        for(int j=0;j<n-i-1;j++){
            if(arr[j]>arr[j+1]){
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

void par_bubble_sort(vector<int> &arr, int n){


    for(int i=0;i<n-1;i++){
        int start=i%2;
        #pragma omp parallel for
        for(int j=start;j<n-1;j+=2){
            if(arr[j]>arr[j+1]){
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

void merge(vector<int> &arr, int l, int m, int r){
    int n1=m-l+1;
    int n2=r-m;
    vector<int> L(n1), R(n2);

    for(int i=0;i<n1;i++){
        L[i]=arr[l+i];
    }

    for(int i=0;i<n2;i++){
        R[i]=arr[m+1+i];
    }

    int i=0;
    int j=0;
    int k=l;
    while(i<n1 && j<n2){
        if(L[i]<=R[j]){
            arr[k]=L[i];
            i++;
        }else{
            arr[k]=R[j];
            j++;
        }
        k++;
    }

    while(i<n1) arr[k++]=L[i++];
    while(j<n2) arr[k++]=R[j++];
}
void seq_merge_sort(vector<int> &arr, int l, int r){
    int n=arr.size();
    if(l<r){
        int m=(l+r)/2;
        seq_merge_sort(arr, l, m);
        seq_merge_sort(arr, m+1, r);
        merge(arr, l, m,r);
    }
}

void par_merge_sort(vector<int> &arr, int l, int r, int depth=0){
    
    if(l<r){
        int m=(l+r)/2;
        if(depth<4){
            #pragma omp parallel sections
            {   
                #pragma omp section
                par_merge_sort(arr, l, m, depth+1);

                #pragma omp section
                par_merge_sort(arr, m+1, r, depth+1);

            }
            
        }else{
            seq_merge_sort(arr, l, m);
            seq_merge_sort(arr, m+1, r);
        }
        
        merge(arr, l, m,r);
    }
}





int main(){


    int n=1000000;
    vector<int> arr(n);
    
    for(int i=0;i<n;i++){
        int x=rand()%n;
        arr[i]=x;
        
    }
    vector<int>v1=arr, v2=arr, v3=arr,v4=arr;

    cout << fixed << setprecision(8);
    // sequential bubble sort
    auto start1=chrono::high_resolution_clock::now();
    seq_bubble_sort(v1, n);
    auto end1=chrono::high_resolution_clock::now();
    chrono::duration<double>seq_bubble_duration=end1-start1;
    cout<<"\nTime required for sequential bubble sort: "<<seq_bubble_duration.count();

    // parallel bubble sort
    auto start2=chrono::high_resolution_clock::now();
    par_bubble_sort(v2, n);
    auto end2=chrono::high_resolution_clock::now();
    chrono::duration<double> par_bubble_duration=end2-start2;
    cout<<"\nTime required for parallel bubble sort: "<<par_bubble_duration.count();

    cout<<"\nSpeedup in sequential and parallel bubble sort: "<<seq_bubble_duration.count()/par_bubble_duration.count();

    // sequential MergeSort
    auto start3=chrono::high_resolution_clock::now();
    seq_merge_sort(v3, 0,n-1);
    auto end3=chrono::high_resolution_clock::now();
    chrono::duration<double> seq_merge_duration=end3-start3;

    cout<<"\nTime required for sequential merge sort: "<<seq_merge_duration.count();

    // parallel merge sort
    auto start4=chrono::high_resolution_clock::now();
    par_merge_sort(v4, 0, n-1);
    auto end4=chrono::high_resolution_clock::now();
    chrono::duration<double> par_merge_duration=end4-start4;

    cout<<"\nTime required for parallel merge sort: "<<par_merge_duration.count();

    cout<<"\nSpeedup in sequential and parallel Merge sort: "<<seq_merge_duration.count()/par_merge_duration.count();
    
}