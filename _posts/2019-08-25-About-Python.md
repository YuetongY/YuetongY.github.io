---
layout: post
title:  "About Python"
subtitle: "关于Python是否是解释型语言的讨论"
author: "yuetong"
header-img: "img/about_python.jpeg"
date:   2019-08-25
tags:
    - Python
    - 编程语言
---
# `pyc`文件
Python是一门解释型的语言，但是在运行之前，需要将`.py`文件中的源码编译为`.pyc`文件，`.pyc`文件存放的是源代码在Python解释器运行之后编译得到的byte code。Python虚拟机之后再执行这些字节码。
根据这个说明，Python程序在执行之前都会生成`.pyc`文件，因为只有byte code才能够被相应的虚拟机所执行。但是，实际上，并不完全正确。
假设现在有两个Python脚本文件`abc.py`和`xyz.py`，其中`abc.py`中将xyz作为一个模块导入其中，换句话说，`abc.py`中有`import xyz`语句。如果执行`python abc.py`，那么只会有`xyz.pyc`文件。
这个例子说明了，运行某个脚本时，python解释器并不会为这个脚本生成`.pyc`文件，只会为这个脚本`import`的一些模块生成`pyc`文件。其实这种文件存在的主要原因是为了加速`Python`程序的编译运行速度，当导入的模块没有发生变化的时候，就不需要再重新生成`pyc`文件了。
在`Python2`中，`pyc`文件通常生成之后会存放在与`py`文件相同的目录下。在`Python3`中，这些文件会放在`__pycache__`目录中。
# Python是“解释型”语言？
在查找关于`pyc`文件的相关内容的时候，关注到有一些关于Python是否是解释型语言的讨论。将讨论的内容总结如下：
- Python是一个语言规范
	通常说Python是一门编程语言，但是网上的资料给出的更加确切的论述是，Python是一种语言规范，我们写Python代码的时候，都是遵循着Python的语言规范形成`py`文件。
- Python有多种实现
	我理解的Python的实现的含义就是，Python的解释器将Python解释为某种字节码，并且在某种特定要求的机器上运行的一种特定环境。
	Python有多种实现，这些实现将Python脚本编译为不同字节码，并交由不同的虚拟环境进行解释执行。举几个常见的例子：
	- CPython：将Python脚本编译为CPython bytecode，然后解释执行
	- Jython：将Python脚本编译为JVML的bytecode，之后交由JVM进行解释执行
	- IronPython：首先将Python脚本解释为CIL bytecode，之后这个字节码如何执行取决于具体的环境，在.NET, GNU Portable.NET 以及 Novell Mono将会将这些字节码编译为相应的机器码执行。
- 任何一门语言都不能简单的确定是否是**解释型语言**
从上面的分析可以看出，一门语言实际上只定义了一种语言规范，并没有指出具体的实现方式。有些实现方式可能将源代码编译为字节码，并交由虚拟机解释执行，有些实现方式可能将源代码编译成为机器码执行。因此，讨论一门语言是解释型还是编译型的语言，是要基于某个语言的具体实现上进行讨论的。
# 参考资料

[How do I create a .pyc file?](http://effbot.org/pyfaq/how-do-i-create-a-pyc-file.htm)

[If Python is interpreted, what are .pyc files?](https://stackoverflow.com/questions/2998215/if-python-is-interpreted-what-are-pyc-files)

[What are Python .pyc files and how are they used?](https://www.quora.com/What-are-Python-pyc-files-and-how-are-they-used)

[where are the .pyc files?](https://stackoverflow.com/questions/5149832/where-are-the-pyc-files)

[What is __pycache__?](https://stackoverflow.com/questions/16869024/what-is-pycache)
