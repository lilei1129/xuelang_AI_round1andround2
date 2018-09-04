飞的更高队复赛赛提交工程文件说明：

操作步骤：

（1）您需要将官网上提供的5个文件解压放入data/official文件夹，分别是：

xuelang_round1_answer_a_20180808，

xuelang_round1_answer_b_20180808，

xuelang_round1_train_part1_20180628，

xuelang_round1_train_part2_20180705，

xuelang_round1_train_part3_20180709,

将第一轮的2个test文件放在test文件夹下：

xuelang_round1_test_a_20180709，

xuelang_round1_test_b;

将第二轮的测试集放在data路径下：

xuelang_round2_test_a_20180809。

（2）使用方法：
　　1 先运行main_0828_V3.py，进行图片预处理工作。

​	2 运行结束之后选择Xception.py或者Xception_1.py进行训练，最后，运行prediect_V3.py进行预测，此时预测得到的是纯11分类的结果。（此处提供一个复杂版本的预测函数，在other_codes文件夹下，名为prediect_512.py）

​	3 如果需要进行2+11版本的预测，首先需要一个二分类结果，然后将二分类的csv文件放在submit文件夹下，然后运行2+11_0829.py即可。

使用过程中注意修改文件路径，改为你的文件路径。

2+11_0829.py进行预测。

​	4 运行结束后在submit文件夹会有预测的csv文件。

（other_codes文件夹下为比赛中用到的所有代码）

==PS：有任何问题，请联系飞的更高队队长（李磊，电话：13545383222，邮箱：1642074995@qq.com)==