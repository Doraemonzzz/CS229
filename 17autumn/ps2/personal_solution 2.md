#### 1. Logistic Regression: Training stability

(a)在数据A上训练logistic regression model很快就收敛了，在数据B上训练logistic regression model无法收敛。

(b)观察后可以发现$\theta$的模长来越大，回顾logistic regression model
$$
h_θ(x) = g(θ^T x) ,g(z) = 1/(1 + e^{−z}),g(z)' = g(z)(1-g(z))
$$
当$\theta$的模长很大时，$\theta^Tx$的模长很大，$g(\theta^T x) \to 0$，$g(z)' = g(z)(1-g(z)) \to 0$，从而梯度会越来越小，训练会很慢。

之所以数据B发生这个现象而A没有发生这个现象，是因为数据A线性不可分，数据B线性可分。

由数据B线性可分可的
$$
y_i\theta^T x_i \ge 0
$$
我们的目标函数为
$$
J(\theta) =-\frac 1 m \sum_{i=1}^m \log (h_{\theta}(y^{(i)}x^{(i)}))
$$
要使得使得目标函数越小，只要$h_{\theta}(y^{(i)}x^{(i)})$越大即可，由于$y_i\theta^T x_i\ge 0$，所以$\theta$的模长越大，$y_i\theta^T x_i$就会越大，由梯度下降的性质可知，迭代之后会让$\theta​$的模长越来越大，就会发生上述现象。

而数据$A$不是线性可分的，所以存在$j​$，使得
$$
y_j\theta^T x_j < 0
$$
所以算法不会让$\theta$的模长不停地增加。

(c)要解决上述问题，关键是不能让$\theta$的模长不停地增长，所以(iii),(v)是最好的方法。

(d)SVM不会发生这个问题，因为SVM是最大间隔分类器，即使可分，最大距离分类器也是唯一的，不会无限迭代下去。

而logistic回归实际上是在让函数间隔变大，所以会出现无法收敛的情形。



#### 2.Model Calibration

(a)只要考虑两个分子即可，logistic回归的输出范围为$(0,1)$，题目中的$(a,b) = (0,1)$，所以
$$
\sum_{i\in I_{a,b}} P (y^{(i)} = 1|x^{(i)} ; θ)= \sum_{i=1}^m  P (y^{(i)} = 1|x^{(i)} ; θ) \\
\sum_{i\in I_{a,b}}1\{y^{(i)} = 1\} =  \sum_{i=1}^m  1\{y^{(i)} = 1\}
$$
接下来证明这两项相等。

回顾损失函数
$$
J(\theta) =-\frac 1 m \sum_{i=1}^m \Big(y^{(i)}\log (h_{\theta}(x^{(i)})) +
(1-y^{(i)})\log (1-h_{\theta}(x^{(i)}))
\Big)\\
y^{(i)} ∈\{0, 1\}, h_θ(x) = g(θ^T x) ,g(z) = 1/(1 + e^{−z})
$$
回顾课本的讲义可得
$$
\frac{\partial} {\partial \theta_j} J(\theta)
=-\frac 1 m \sum_{i=1}^m (y^{(i)}- h_{\theta}(x^{(i)})) x_j
$$
那么
$$
\nabla_{\theta} J(\theta) = -\frac 1 m  X^T S
$$
其中
$$
x_k=[1,x_1^{(k)},...,x_n^{(k)}]^T \in \mathbb  R^{n+1}\\
  X=  \left[
 \begin{matrix}
 x_1^T\\
  ...\\
x_m^T 
  \end{matrix}
  \right] =
   \left[
 \begin{matrix}
 1 &x_1^{(1)} & ... &x_n^{(1)}\\
  ...&...& ... &...\\
 1 &x_1^{(m)} & ... &x_n^{(m)}
  \end{matrix}
  \right] 
  \in  \mathbb R^{m\times (n+1)} \\
  S = \left[
 \begin{matrix}
  y^{(1)}-h_θ(x^{(1)}) \\
... \\
  y^{(m)}-h_θ(x^{(m)})
  \end{matrix}
  \right] \in \mathbb  R^m
$$
由$\theta​$的选择规则可知
$$
X^T S =0 
$$
这里有$n+1$个等式，注意$X^T$的第一行全为$1$，所以我们考虑第一个等式
$$
[1,...,1] S = 0 \\
 \sum_{i=1}^m  y^{(i)}-h_θ(x^{(i)}) = 0\\
  \sum_{i=1}^m y^{(i)}  =   \sum_{i=1}^mh_θ(x^{(i)})
$$
由于$y^{(i)} \in \{0,1\}, h_θ(x^{(i)}) = P (y^{(i)} = 1|x^{(i)} ; θ)$，所以上式即为
$$
\sum_{i=1}^m  P (y^{(i)} = 1|x^{(i)} ; θ) =   \sum_{i=1}^m  1\{y^{(i)} = 1\}
$$
从而
$$
\sum_{i\in I_{a,b}} P (y^{(i)} = 1|x^{(i)} ; θ)=\sum_{i\in I_{a,b}}1\{y^{(i)} = 1\}
$$
命题得证。

(b)考虑两个数据的数据集$x^{(1)},x^{(2)}$，不妨设$y^{(1)}=1,y^{(2)}=0$，如果
$$
P (y^{(1)} = 1|x^{(1)} ; θ) =0.4,  P (y^{(2)} = 1|x^{(2)} ; θ) =0.6
$$
那么我们预测$y^{(1)} = 0,y^{(2)} = 1$，准确率为$0$，但是
$$
\sum_{i\in I_{a,b}} P (y^{(i)} = 1|x^{(i)} ; θ)=\sum_{i\in I_{a,b}}1\{y^{(i)} = 1\}= 1
$$
所以perfectly calibrated无法推出perfect accuracy。

反之，由(a)可知必然成立。


(c)设损失函数为
$$
J(\theta) =-\frac 1 m \sum_{i=1}^m \Big(y^{(i)}\log (h_{\theta}(x^{(i)})) +
(1-y^{(i)})\log (1-h_{\theta}(x^{(i)}))
\Big) + C \sum_{i=1} ^{n+1} \theta_i^2
$$
记
$$
\theta = \left[
 \begin{matrix}
  \theta_1\\
... \\
 \theta_{n+1}
  \end{matrix}
  \right]
$$
继续使用(a)的记号，那么
$$
\nabla J(\theta) = -\frac 1  m X^T S +2C \theta  =0 \\
X^T S  = 2m C\theta 
$$
依旧考虑第一个等式
$$
[1,...,1] S =  2mC \theta_1 \\ 
 \sum_{i=1}^m y^{(i)}-h_θ(x^{(i)}) = 2mC \theta_1 \\ 
   \sum_{i=1}^m y^{(i)}  =   \sum_{i=1}^mh_θ(x^{(i)})+ 2mC\theta_1 \\
    \sum_{i=1}^m  1\{y^{(i)} = 1\} = \sum_{i=1}^m  P (y^{(i)} = 1|x^{(i)} ; θ)  +  2mC\theta_1
$$
从而
$$
\sum_{i=1}^m  1\{y^{(i)} = 1\} = \sum_{i=1}^m  P (y^{(i)} = 1|x^{(i)} ; θ)  +  2mC\theta_1
$$


#### 3.Bayesian Logistic Regression and weight decay

回顾定义
$$
\begin{aligned}
θ_{\text{MAP}} 
&= \arg \max_θ p(θ) \prod _{i=1}^m p(y^{(i)}|x^{(i)}, θ)  \\
&=  \arg \max_θ  \exp({- \frac{||\theta||^2}{2 \tau ^2}} ) \prod _{i=1}^m p(y^{(i)}|x^{(i)}, θ)
\end{aligned}
$$
由定义可知
$$
\exp({- \frac{||θ_{\text{MAP}}||^2}{2 \tau ^2}} ) \prod _{i=1}^m p(y^{(i)}|x^{(i)}, θ_{\text{MAP}}) \ge  \exp({- \frac{||θ_{\text{ML}}||^2}{2 \tau ^2}} ) \prod _{i=1}^m p(y^{(i)}|x^{(i)},θ_{\text{ML}})
$$
因为
$$
\prod _{i=1}^m p(y^{(i)}|x^{(i)}, θ_{\text{MAP}})\le  \prod _{i=1}^m p(y^{(i)}|x^{(i)},θ_{\text{ML}})
$$
所以
$$
\exp({- \frac{||θ_{\text{MAP}}||^2}{2 \tau ^2}} )  \ge  \exp({- \frac{||θ_{\text{ML}}||^2}{2 \tau ^2}} ) 
$$
从而
$$
||θ_{\text{MAP}}||_2 \le ||θ_{\text{ML}}||_2
$$



#### 4.Constructing kernels

假设$K_i$对应的矩阵为$M_i$，$K$对应矩阵为$M$，由核函数的定义可知$M_i $为半正定阵。

(a)$K(x,z)= K_1(x,z) + K_2(x,z) $是核，因为
$$
\begin{aligned}
x^TMx&=x^T(M_1+M_2)x
\\&=x^TM_1x+x^TM_2x
\\&\ge0
\end{aligned}
$$
(b)$K(x,z)= K_1(x,z) -K_2(x,z) $不是核。取$K_2(x,z)=2K_1(x,z)$
$$
x^TMx=x^T(M_1-M_2)x=-x^TM_1x\le0
$$
(c)$K(x,z) = aK_1(x,z) ,a>0$是核
$$
\begin{aligned}
x^TMx&=x^T(aM_1)x
\\&=ax^TMx
\\&\ge0
\end{aligned}
$$
(d)$K(x,z) = -aK_1(x,z) ,a>0​$不是核
$$
\begin{aligned}
x^TMx&=x^T(-aM_1)x
\\&=-ax^TMx
\\&\le0
\end{aligned}
$$
(e)$K(x,z) = K_1(x,z) K_2(x,z)$是核

因为$K_1,K_2$为核，所以设$K_1(x,z) = Φ_1(x) Φ^T_1(z),K_2=Φ_2(x) Φ^T_2(z)$。

记$\Phi(x)$是$Φ_1(x)Φ_2^T(x)$每一行拼接而成的向量，设$Φ_1(x),Φ_2(x)\in \mathbb R^n$，给出以下记号
$$
\Phi^i(x)=Φ_1^{i}(x)Φ^T_2(x)\in\mathbb  R^{1\times n}\\
Φ^{i}_1(x)为Φ_1(x)的第i个分量
$$
那么
$$
\begin{aligned}
\Phi(x) =
 \left[
 \begin{matrix}
   Φ^1(x) &
   Φ^2(x)&
...&
     Φ^n(x)
  \end{matrix}
  \right] \in \mathbb R^{1\times n^2}
\end{aligned}
$$
接着计算$\Phi(x) \Phi^T(x') ​$，注意$\Phi^i(x)​$为行向量
$$
\begin{aligned}
(Φ(x) Φ^T(x^{'}))
&=\sum_{i=1}^n  (Φ^i(x))Φ^i(x^{'})^T\\
&=\sum_{i=1}^n (Φ_1^{i}(x)Φ_2(x)^T)(Φ_1^{i}(x^{'})Φ_2(x^{'})^T)^T\\
&=\sum_{i=1}^nΦ_1^{i}(x)Φ_1^{i}(x^{'})Φ_2(x)^TΦ_2(x^{'})\\
&=\sum_{i=1}^nΦ_1^{i}(x)Φ_1^{i}(x^{'})K_2(x,x^{'})\\
&=K_2(x,x^{'})\sum_{i=1}^nΦ_1^{i}(x)Φ_1^{i}(x^{'})\\
&=K_2(x,x^{'})K_1(x,x^{'})
\end{aligned}
$$
所以$Φ(x)$对应的核为$K_1(x,x^{'})K_2(x,x^{'})$，从而$K(x,z) = K_1(x,z) K_2(x,z)$是核。

(f)$K(x,z) = f(x)f(z)$是核，因为符合定义。

(g)$K(x,z) = K_3(\phi(x), \phi(z))$是核，因为
$$
y^T M y =y^T M_3 y \ge 0
$$
(h)由(e)可知，如果$K_1$是核，那么$K_1^i (i\ge1, i\in N)$也是核，又由(a)(c)可得核函数的正系数的线性组合为核，所以$K(x,z)=p(K_1(x,z))$也是核。



#### 5.Kernelizing the Perceptron

设这里的数据为$x_1,...,x_m$

(a)根据更新公式
$$
θ^{(i+1)} := θ^{(i)} + α1\{g({θ^{(i)}}^T \phi(x^{(i+1)}))y^{(i+1)} < 0\}y^{(i+1)}\phi(x^{(i+1)})
$$
如果初始化$\theta^{(0)}=0​$，那么$\theta^{(i)}​$可以表示为$\phi(x^{(i)})​$的线性组合，从而
$$
\theta^{(i)} = \sum_{j=1}^m \beta^{(j)} \phi(x^{(j)})
$$
(b)计算$g({θ^{(i)}}^T \phi(x^{(i+1)}))​$
$$
g({θ^{(i)}}^T \phi(x^{(i+1)})) = g\Big( \sum_{j=1}^m \beta^{(j)}  \phi(x^{(j)})^T \phi(x^{(i+1)}) \Big) = g\Big( \sum_{j=1}^m \beta^{(j)}  M_{j,i+1}  \Big)
$$
 (c)由上述公式可知，我们只要更新$\beta^{(i)} =(\beta_1^{(i)},...,\beta_m^{(i)})​$即可，更新公式如下
$$
记第i+1轮选择的数据为x^{(i+1)}，对应的顺序为a_{i+1}\\
\beta^{(i+1)}:= \beta^{(i)} +\alpha 1\{ g({θ^{(i)}}^T \phi(x^{(i+1)})) y^{(i+1)} <0\} y^{(i+1)} \underbrace{ (0,...,1,...0)^T}_ {第a_{i+1}项为1，其余为0}
$$



#### 6.Spam classication

见代码

