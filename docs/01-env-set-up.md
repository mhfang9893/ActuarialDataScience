# 准备工作 {#ch1}

## 克隆代码

[GitHub](https://github.com/)提供了大量开源代码，这门课的代码主要来自[此链接](https://github.com/JSchelldorfer/ActuarialDataScience)。通常，使用GitHub开源代码最方便的是`fork`到自己GitHub账户下，然后`clone`到本地。具体而言，需要进行以下操作：

1. 注册GitHub账户。

2. `Fork`[此链接](https://github.com/JSchelldorfer/ActuarialDataScience)到自己账户下的新仓库,可重新命名为如`Modern-Actuarial-Models`。

3. 安装[git](https://git-scm.com/)。在命令窗口使用`$ git config --global user.name "Your Name"` 和 `$ git config --global user.email "youremail@yourdomain.com"` 配置git的用户名和邮箱分别为GitHub账户的用户名和邮箱。最后可使用`$ git config --list`查看配置信息。

4. 在本地电脑创建ssh public key，并拷贝到GitHub中个人设置中，ssh public key一般保存在本人目录下的隐藏文件夹ssh中。详见[链接](https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/connecting-to-github-with-ssh)。

5. 电脑连接手机4G热点。这步主要是为了加速下部克隆的速度。

6. 在RStudio中创建新的项目，选择Version Control，然后Git，在Repository URL中输入GitHub中刚才建立的新仓库地址（在`Code`下能找到克隆地址，建议使用SSH地址，可以避免后续`push`代码到云端时，每次都需要输入密码的麻烦），输入文件夹名称，选择存放位置，点击`create project`，R开始下载GitHub上该仓库的所有内容。 

7. 此时，你在GitHub上仓库的内容全部克隆到了本地，且放在了一个R Project中。在该Project中，会多两个文件，.Rproj和.gitignore，第一个文件保存了Project的设置，第二文件告诉git在`push`本地文件到GitHub时哪些文件被忽略。

8. 如果你修改了本地文件，可以通过R中内嵌的Git上传到GitHub（先`commit`再`push`），这样方便在不同电脑上同步文件。git是代码版本控制工具，在`push`之前，你可以比较和上个代码版本的差异。GitHub记录了你每次`push`的详细信息，且存放在本地文件夹.git中。

9. (选做) 你可以建立新的`branch`，使自己的修改和源代码分开。具体操作可参考[链接](https://resources.github.com/whitepapers/github-and-rstudio/)，或者参考账户建立时自动产生的`getting-started`仓库。

10. (选做) 你可以尝试[Github Desktop](https://desktop.github.com/)或者[Jupyter Lab](https://jupyter.org/)（加载git extension）管理，但对于这门课，这两种方式不是最优。

理论上，GitHub上所有仓库都可以采用以上方法在RStudio中管理，当然，RStudio对于R代码仓库管理最有效，因为我们可以直接在RStudio中运行仓库中的代码。

## 建立环境

此门课程用到了R和python，大家对于R应该很熟悉，python也应该在机器学习的课程中有所接触。此外，这门课程还用到了`interface to tensorflow`，即在R中调用python中的`tensorflow`。

### R

`reticulate`是interface to python。

1. 安装

2. 可以让R自动安装

3. 指定tensorflow的目录




### Python

1. 安装Anaconda。

2. 建立独立环境`conda create env -n "env-name" python=3.8 tensorflow notebook`。

3. 激活环境`conda activate "env-name"`.

4. cd 到你的工作目录。

5. 启动jupyter notebook `jupyter notebook`。

6. 如遇到缺少的包，在该环境下conda安装。





