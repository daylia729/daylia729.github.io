# 小李的知识库

## 如何配置

1. 下载 git

   [【『教程』简单明了的Git入门】](https://www.bilibili.com/video/BV1Cr4y1J7iQ/?share_source=copy_web&vd_source=54b55542474b885b589dd23e8edb6b98)

2. 把本项目下载到本地

   在本地选取一个新的文件夹，打开 git bash 或终端，输入以下命令：

   ```bash
   git clone https://github.com/daylia729/daylia729.github.io.git
   ```

   此时，本地就会多一个名为 `daylia729.github.io` 的文件夹，里面就是本项目的所有代码。

3. 配置环境与启动项目

   ```bash
   cd daylia729.github.io   # 进入项目文件夹
   npm install              # 安装依赖
   npm run dev              # 启动项目
   ```

4. 迁移博客内容

    将原有的博客文章与项目配置文章迁移过来。

5. 提交代码更新

   ```bash
   git add .               # 添加所有文件到暂存区
   git commit -m "update"  # 提交更新
   git push                # 推送更新到远程仓库
   ```

此时，本地更改的内容就可以同步到云端啦。

同时网页内容也会随之更新：

[https://daylia729.github.io/](https://daylia729.github.io/)
