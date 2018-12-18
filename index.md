### <a name="dl-tracker"> 实时视频目标跟踪
http://www.robots.ox.ac.uk/~luca/siamese-fc.html   
https://github.com/torrvision/siamfc-tf   
http://davheld.github.io/GOTURN/GOTURN.pdf   
 
### <a name="it-parted-big-disk"> linux 2T以上硬盘分区 
parted /dev/sdb mklabel gpt  
parted /dev/sdb mkpart primary 0 4000000 创建4T的分区  
mkfs -t ext4 /dev/sdb1  
mount /dev/sdb1 /mnt/sdb   


### <a name="code-build-python-to-exe"> 编译python程序到可执行文件
https://github.com/Nuitka/Nuitka  
https://zhuanlan.zhihu.com/p/31721250  
https://blog.526net.com/?p=3166  


### <a name="it-nfs">ubuntu16.04 安装nfs</a>
服务器端  
sudo apt install nfs-kernel-server  
 编辑/etc/exports 文件，export /logs目录  
/logs *(rw,sync,no_subtree_check,no_root_squash)  
重启nfs服务使配置生效    
sudo service nfs-kernel-server restart  
客户端   
sudo apt install nfs-common  
编辑/etc/fstab 自动挂载服务器目录到本地目录  
192.168.1.234:/logs      /mnt/logs      nfs     defaults         0      2  
sudo mount -a

### <a name="dl-caffelog">关闭cafffe日志输出</a>
export GLOG_minloglevel=1  

### <a name="it-dnsserver">ubuntun linux 域名服务器配置</a>
vim /etc/resolvconf/resolv.conf.d/base  
nameserver 8.8.8.8   
sudo resolvconf -u  

### <a name="it-routeconf">ubuntu linux 路由表配置</a>
所有发往192.168.*.*通过192.168.11.1  
sudo route add -net 192.168.0.0 netmask 255.255.0.0 gw 192.168.11.1  

路由配置加到 /etc/rc.local 避免重启后丢失  
route add -net 172.16.0.0 netmask 255.255.0.0 gw 172.16.0.1  

### <a name="it-ignorenetwork">ubuntu跳过开机网络检测</a>
进入系统后修改文件/etc/systemd/system/network-online.target.wants/networking.service  
命令：sudo vim /etc/systemd/system/network-online.target.wants/networking.service  
将里面的TimeoutStartSec=5min 修改为TimeoutStartSec=2sec  

### <a name="it-ramfs">linux 建立100M的ramfs</a>
sudo mount -t ramfs none ramfs -o maxsize=100000


### <a name="dl-mscoco2017">MS coco 2017 数据集</a>
http://images.cocodataset.org/zips/train2017.zip  
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip 
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

http://images.cocodataset.org/zips/test2017.zip 
http://images.cocodataset.org/annotations/image_info_test2017.zip 
这些就是全部的microsoft coco数据集2017的链接了。

在linux上可以使用axel多线程工具来下载

### <a name="code-excel-vba-mysql">Excel使用VBA与访问MySql数据</a>
* 安装mysql odbc 32位驱动 https://dev.mysql.com/downloads/connector/odbc/
* 打开Excel按下Alt+F11打开VBE
* 在VBE菜单栏选择“工具”－“引用”，在弹出的引用窗口中，找到"Microsoft ActiveX Data Objects 6.1 Library"和"Microsoft ActiveX Data Objects Recordset 6.0 Library"，把前面的框勾选上，点击确定即可
```vb
    Dim conn As ADODB.Connection
    Dim rs As ADODB.Recordset
    Set conn = New ADODB.Connection
    Set rs = New ADODB.Recordset
    ' 一定要安装mysql odbc 32位驱动 https://dev.mysql.com/downloads/connector/odbc/
    conn.ConnectionString = "Driver={MySQL ODBC 8.0 Unicode Driver};Server=192.168.1.252;DB=huanbao;UID=huanbao;PWD=huanbao;OPTION=3;"
    conn.Open
    rs.Open "select * from `HB_BJJL`", conn
    'copy数据记录到excel表格
    Range("A2").CopyFromRecordset rs
    rs.Close: Set rs = Nothing
    conn.Close: Set conn = Nothing
```

### <a name="it-nginx-ERR_INCOMPLETE_CHUNKED_ENCODING">net::ERR_INCOMPLETE_CHUNKED_ENCODING 错误</a>

net::ERR_INCOMPLETE_CHUNKED_ENCODING 错误，但是本地调试没有这个错误，最终发现是服务器的Nginx 配置上有问题，查看error.log 显示的是Upstream prematurely closed connection while reading upstream...错误，翻墙谷歌最终在nginx.conf 的http模块内加入


proxy_request_buffering off;  
proxy_buffering off;  


### [imdb和roidb的说明](./rcnn-roidb)

### [2018-10-17 古建筑笔记](./gujian-note)
