# 准备工作 {#ch1}

## 克隆代码

[GitHub](https://github.com/)提供了大量开源代码，这门课的代码主要来自[此链接](https://github.com/JSchelldorfer/ActuarialDataScience)。通常，使用GitHub开源代码最方便的是`fork`到自己GitHub账户下，然后`clone`到本地。具体而言，需要进行以下操作：

1. 注册GitHub账户。

2. `Fork`[此链接](https://github.com/JSchelldorfer/ActuarialDataScience)到自己账户下的新仓库,可重新命名为如`Modern-Actuarial-Models`。

3. 安装[git](https://git-scm.com/)。在命令窗口使用`$ git config --global user.name "Your Name"` 和 `$ git config --global user.email "youremail@yourdomain.com"` 配置git的用户名和邮箱分别为GitHub账户的用户名和邮箱。最后可使用`$ git config --list`查看配置信息。

4. 在本地电脑创建ssh public key，并拷贝到GitHub中个人设置中，ssh public key一般保存在本人目录下的隐藏文件夹ssh中。详见[链接](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/connecting-to-github-with-ssh)。

5. 电脑连接手机4G热点。这步主要是为了加速下步克隆的速度。

6. 在RStudio中创建新的项目，选择Version Control，然后Git，在Repository URL中输入GitHub中刚才建立的新仓库地址（在`Code`下能找到克隆地址，建议使用SSH地址，可以避免后续`push`代码到云端时，每次都需要输入密码的麻烦），输入文件夹名称，选择存放位置，点击`create project`，R开始下载GitHub上该仓库的所有内容。 

7. 此时，你在GitHub上仓库的内容全部克隆到了本地，且放在了一个R Project中。在该Project中，会多两个文件，.Rproj和.gitignore，第一个文件保存了Project的设置，第二文件告诉git在`push`本地文件到GitHub时哪些文件被忽略。

8. 如果你修改了本地文件，可以通过R中内嵌的Git上传到GitHub（先`commit`再`push`），这样方便在不同电脑上同步文件。git是代码版本控制工具，在`push`之前，你可以比较和上个代码版本的差异。GitHub记录了你每次`push`的详细信息，且存放在本地文件夹.git中。

9. (选做) 你可以建立新的`branch`，使自己的修改和源代码分开。具体操作可参考[链接](https://resources.github.com/whitepapers/github-and-rstudio/)，或者参考账户建立时自动产生的`getting-started`仓库。

10. (选做) 你可以尝试[Github Desktop](https://desktop.github.com/)或者[Jupyter Lab](https://jupyter.org/)（加载git extension）管理，但对于这门课，这两种方式不是最优。

理论上，GitHub上所有仓库都可以采用以上方法在RStudio中管理，当然，RStudio对于R代码仓库管理最有效，因为我们可以直接在RStudio中运行仓库中的代码。

## 建立环境

首先下载并安装[Anaconda](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)或者[Miniconda](https://docs.conda.io/en/latest/miniconda.html)，并通过修改用户目录下的`.condarc`文件使用[TUNA镜像源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/). 这步很关键，否则下面的安装会很慢。

### R interface to python

安装R包`reticulate`，它可以建立R与python的交互。常见的命令如下：

- `conda_list()`列出已安装的conda环境

- `virtualenv_list()`列出已存在的虚拟环境

- `use_python, use_conda, use_virtualenv`可以指定与R关联的python。

- `py_config()`可以查看当前python关联信息。

很多时候，R会创建一个独立conda环境`r-miniconda/envs/r-reticulate`。

### R

这里主要说明`keras`包的安装和使用。[Keras](https://keras.rstudio.com/)是tensorflow的API，在keras中建立的神经网络模型都由tensorflow训练。安装`keras`包主要是安装python库tensorflow，并让R与之相关联。

1. `install.packages("tensorflow")`。

2. 如果未装tensorflow库则运行`install_tensorflow()`，该命令会自动选择合适的方法安装tensorflow；如果本地已经安装tensorflow库，可以使用命令`reticulate:use_conda("your_tensorflow_env")`关联`your_tensorflow_env`。

### Python

1. 建立独立环境`conda create env -n "env-name" python=3.8 tensorflow notebook`。

2. 激活环境`conda activate "env-name"`.

3. cd 到你的工作目录。

4. 启动jupyter notebook `jupyter notebook`。

5. 如遇到缺少的包，在该环境`env-name`下使用conda安装缺少的包。






