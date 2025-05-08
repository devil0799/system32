#include<bits/stdc++.h>
#include<omp.h>
#include<chrono>
using namespace std;
using namespace std::chrono;


void seq_linear_regression(vector<double>&x , vector<double>&y){
    //sequential
    cout<<"\nSequnetial:";
    double sum_x=0.0,sum_xy=0.0,sum_y=0.0,sum_x2=0.0;
    int n = x.size();
    for(int i=0;i<n;i++){
        sum_x+=x[i];
        sum_y+=y[i];
        sum_xy+=x[i]*y[i];
        sum_x2+=x[i]*x[i];
    }
    double m = (n*sum_xy-(sum_x*sum_y))/((n*sum_x2)-(sum_x*sum_x));
    cout<<"\nSlope(Sequential):"<<m;
    double c = (sum_y-(m*sum_x))/n;
    cout<<"\nConstant(Sequential):"<<c;
    cout<<endl;

}

void par_linear_regression(vector<double>&x , vector<double>&y){
    cout<<"\nParallel:";
    double sum_x=0.0,sum_xy=0.0,sum_y=0.0,sum_x2=0.0;
    int n = x.size();
    #pragma omp parallel for reduction(+ : sum_x,sum_y,sum_xy,sum_x2)
    for(int i=0;i<n;i++){
        sum_x+=x[i];
        sum_y+=y[i];
        sum_xy+=x[i]*y[i];
        sum_x2+=x[i]*x[i];
    }
    double m = (n*sum_xy-(sum_x*sum_y))/((n*sum_x2)-(sum_x*sum_x));
    cout<<"\nSlope(Parallel):"<<m;
    double c = (sum_y-(m*sum_x))/n;
    cout<<"\nConstant(Parallel):"<<c;
    cout<<endl;
}

int main(){
    int size;
    cout<<"\nEnter Points Size:";
    cin>>size;
    vector<double>X(size);
    vector<double>Y(size);

    for(int i=0;i<size;i++){
        X[i] = rand()%1000;
        Y[i] = X[i]*3+7+rand()%50;
    }

    auto t1 = high_resolution_clock::now();
    seq_linear_regression(X,Y);
    auto t2 = high_resolution_clock::now();

    auto t3 = high_resolution_clock::now();
    par_linear_regression(X,Y);
    auto t4 = high_resolution_clock::now();


    cout<<"\n\nSequential_linear_regresion:"<<(duration_cast<milliseconds>(t2-t1)).count()<<"ms";
    cout<<"\nParallel_linear_regresion:"<<(duration_cast<milliseconds>(t4-t3)).count()<<"ms";
    




}
