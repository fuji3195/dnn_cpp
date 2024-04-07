#include "main.hpp"
#include <fstream>
#include <istream>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <random>
#include <set>
#include "main.hpp"

std::random_device rd;
std::mt19937 gen(rd());

double learning_rate = 0.1;
int batch_size = 128;
int epoch_num = 20;
double eps = 1e-10;

VVD XWplusB(const VVD& X, const VVD& W, const VD B) {
    VVD res(X.size(),VD(W[0].size(),0));
    rep(i,X.size()) rep(j,W.size()) rep(k,W[j].size()) res[i][k] += X[i][j] * W[j][k];
    rep(i,X.size()) rep(j,B.size()) res[i][j] += B[j];
    return res;
}

enum class ACT { ReLU, Softmax, Sigmoid };
enum class OPT { SGD, Momentum, Adam};
class Optimizer {
private:
    int in, out;
    double lr = learning_rate;
    double momentum_rate = 0.9;
    double beta1 = 0.9;
    double beta2 = 0.999;
    VVD mW;
    VVD vW;
    VD  mB;
    VD  vB;
    int t;
    OPT optimizer = OPT::Momentum;
public:
    Optimizer(int in, int out)
    : in(in), out(out), mW(in,VD(out,0.0)), vW(in,VD(out,0.0)), mB(out,0.0), vB(out,0.0), t(0) { }
    void change_optimizer(OPT optimizer) {this->optimizer = optimizer;}
    void update(VVD& w, const VVD& dW, VD& b, const VD& db) {
        switch (optimizer) {
            case OPT::SGD : update_SGD(w,dW,b,db); break;
            case OPT::Momentum: update_Momentum(w,dW,b,db); break;
            case OPT::Adam: update_Adam(w,dW,b,db); break;
        }
    }
    void update_SGD (VVD& w, const VVD& dW, VD& b, const VD& db) {
        rep(i,w.size()) rep(o,w[i].size())  w[i][o] -= lr * dW[i][o];
        rep(o,b.size()) b[o] -= lr * db[o];
    }
    void update_Momentum(VVD& w, const VVD& dW, VD& b, const VD& db) {
        rep(i,w.size()) rep(o,w[i].size()) {
            vW[i][o] = momentum_rate * vW[i][o] - lr * dW[i][o];
            w[i][o] += vW[i][o];
        }
        rep(o,b.size()) {
            vB[o] = momentum_rate * vB[o] - lr * db[o];
            b[o] += vB[o];
        }
    }
    void update_Adam(VVD& w, const VVD& dW, VD& b, const VD& db) {
        ++t;
        double lr_t = lr * sqrt(1.0 - pow(beta2, t)) / (1.0 - pow(beta1, t));
        rep(i,w.size()) rep(o,w[i].size()) {
            mW[i][o] = beta1 * mW[i][o] + (1.0 - beta1) * dW[i][o];
            vW[i][o] = beta2 * vW[i][o] + (1.0 - beta2) * dW[i][o] * dW[i][o];
            w [i][o]-= lr_t  * mW[i][o] / (sqrt(vW[i][o]) + eps);
        }
        rep(o,b.size()) {
            mB[o] = beta1 * mB[o] + (1.0 - beta1) * db[o];
            vB[o] = beta2 * vB[o] + (1.0 - beta2) * db[o] * db[o];
            b [o]-= lr_t  * mB[o] / (sqrt(vB[o]) + eps);
        }
    }
};

class Layer {
private:
    int in, out;
    ACT activation;
    VVD W;
    VD B;
    VVD input_matrix;
    VVD activation_input;
    Optimizer opt;
public:
    Layer() : in(1), out(1), activation(ACT::ReLU), opt(1,1) {}
    Layer(int in, int out, ACT activation) : in(in), out(out), activation(activation), W(in, VD(out)), B(out), opt(in,out) {
        std::normal_distribution<> dist(0,1);
        rep(i,in) rep(o,out) W[i][o] = dist(gen);
        rep(o,out) B[o] = dist(gen);
    }
    ~Layer() {}
    VVD sigmoid(const VVD& x) {
        VVD res(x.size());
        rep(i,x.size()) rep(j,x[i].size()) res[i].push_back(1.0 / (1.0 + exp(-x[i][j])));
        return res;
    }

    VVD softmax(const VVD& input) {
        VVD res(input.size());
        rep(d,input.size()) {
            double sum = 0;
            for (auto val : input[d]) sum += exp(val);
            for (auto val : input[d]) res[d].push_back(exp(val) /sum);
        }
        return res;
    }
    VVD forward(const VVD& input) {
        input_matrix = input;
        auto out = XWplusB(input, W, B);
        activation_input = out;
        switch (activation) {
            case ACT::Sigmoid   : out = sigmoid(out); activation_input = out; break;
            case ACT::Softmax   : out = softmax(out); break;
            default             : ;
        }
        return out;
    }

    VVD sigmoid_backward(const VVD& dldy) {
        auto res = dldy;
        rep(s,dldy.size()) rep(o,out) res[s][o] = dldy[s][o] * activation_input[s][o] * (1 - activation_input[s][o]);
        return res;
    }
    VVD softmax_backward(const VVD& dldy) { return dldy; }
    void update(const VVD& dldw, const VD& dldb, double lr) {
        int Update_Algo = 0;    // SGD
        switch (Update_Algo) {
            case 0 :
                rep(o,out) B[o] -= dldb[o] * lr;
                rep(i,in) rep(o,out) W[i][o] -= dldw[i][o] * lr;
                break;
            case 1 : 
                rep(o,out) B[o] -= dldb[o] * lr;
                rep(i,in) rep(o,out) W[i][o] -= dldw[i][o] * lr;
                break;
            default : ;
        }
    }
    VVD backward(const VVD& dldy, double lr) {
        VVD back;
        int sz = dldy.size();
        switch (activation) {
            case ACT::Sigmoid : back = sigmoid_backward(dldy); break;
            case ACT::Softmax : back = softmax_backward(dldy); break;
            default : back = dldy;
        }
        // bias backward
        VD dldb(out,0); rep(s,sz) rep(o,out) dldb[o] += dldy[s][o];
        rep(o,out) dldb[o] /= sz;
        // weight backward + dldx calc
        VVD dldw(in, VD(out,0)), dldx(sz, VD(in));
        rep(s,sz) rep(i,in) rep(o,out) {
            dldw[i][o] += input_matrix[s][i] * dldy[s][o];
            dldx[s][i] += dldy[s][o] * W[i][o];
        }
        rep(i,in) rep(o,out) dldw[i][o] /= sz;
        // feedback value
        opt.update(W,dldw,B,dldb);
        return dldx; 
    }
};

class NeuralNetwork {
private:
    int in, out;
    vector<Layer> dense;
    int now;
public:
    NeuralNetwork(int input, int output) : in(input), out(output), now(input) { }

    void add_Dense(int hidden, ACT activation) {
        dense.push_back(Layer(now,hidden, activation));
        now = hidden;
    }
    void Set_Loss_Func() {
        dense.push_back(Layer(now,out, ACT::Softmax));
    }

    double Mean_Cross_Entropy_Error(const VVD& ans, const VVD& pred) {
        double val = 0;
        double delta = 1e-10;
        rep(i,ans.size()) rep(j,ans[i].size()) val -= ans[i][j] * log(pred[i][j] + delta);
        return val / ans.size();
    }

    VD fit(const VVD& x, const VVD& y, int batch, double lr, int epoch) {
        std::uniform_int_distribution<int> distr(0,x.size()-1);
        VD history(epoch,0);
        int iter = std::max((int)y.size()/batch,1);
        rep(i,epoch) {
            rep(c,iter) {
                std::set<int> st;
                VVD batch_x(batch), batch_y(batch);
                int cnt = 0;
                while (cnt<batch) {
                    int now = distr(gen);
                    if (st.find(now)!=st.end()) continue;
                    st.insert(now);
                    batch_x[cnt] = x[now];
                    batch_y[cnt] = y[now];
                    ++cnt;
                }
                VVD pred = batch_x;
                rep(d,dense.size()) pred = dense[d].forward(pred);
                VVD err = pred;
                rep(b,batch) rep(o,batch_y[b].size()) err[b][o] -= batch_y[b][o];
                for (int d = dense.size()-1; d>=0; --d) err = dense[d].backward(err, lr);
            }
            VVD pred = x;
            rep(d,dense.size()) pred = dense[d].forward(pred);
            history[i] = Mean_Cross_Entropy_Error(y, pred);
        }
        return history;
    }
    VVD predict(const VVD& x) {
        auto pred = x;
        rep(d,dense.size()) pred = dense[d].forward(pred);
        return pred;
    } 
};


void step2(void) {
    auto tr = get_mnist_data("data/train/data.bin");
    auto ts = get_mnist_data("data/test/data.bin");
    VVD tr_x = tr.first, tr_y = tr.second;
    VVD ts_x = ts.first, ts_y = ts.second;
    cout << "tr_x, tr_y size = " << tr_x.size() <<"," << tr_y.size() << ", ts_x, ts_y size = " << ts_x.size() << "," << ts_y.size() << endl;

    NeuralNetwork net(tr_x.front().size(), tr_y.front().size());
    net.add_Dense(100, ACT::Sigmoid);
    net.Set_Loss_Func();
    auto history = net.fit(tr_x, tr_y, batch_size, learning_rate, epoch_num);
    rep(c,history.size()) cout << "cnt = " << c << ", loss = " << history[c] << endl;
    auto t2 = net.predict(tr_x);
    int t2_cnt = 0;
    rep(i,tr_y.size()) {
        int ans = 0; double exp = -999;
        rep(j,10) if (exp<t2[i][j]) ans = j, exp = t2[i][j];
        if (tr_y[i][ans]>0.5) ++t2_cnt;
    }
    cout << "t2 right ans = " << t2_cnt << " / " << t2.size() << endl;
    
    auto t3 = net.predict(ts_x);
    int t3_cnt = 0;
    rep(i,ts_y.size()) {
        int ans = 0; double exp = -999;
        rep(j,10) if (exp<t3[i][j]) ans = j, exp = t3[i][j];
        if (ts_y[i][ans]>0.5) ++t3_cnt;
    }
    cout << "t3 right ans = " << t3_cnt << " / " << t3.size() << endl;
}

int main(void) {
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
    step2();
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    cout << "elapsed time = " << elapsed << " ms" << endl;
}