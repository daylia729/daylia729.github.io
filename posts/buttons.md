---
title: 一些好看的按钮CSS
description: css
date: 2025-02-26
tags:
  - css
---
## 悬停效果
```CSS
<style>
    .SUBSCRIBE-button{
        background-color:rgb(201, 0, 0);
        color: white;
        border: none;
        border-radius: 2px;
        cursor: pointer;             
        transition: opacity 0.15s;
        padding-left: 16px;
        padding-right: 16px;
        padding-bottom: 10px;
        padding-top: 10px;
        margin-left: 30px;
        margin-right: 10px;
        vertical-align: top;        
    }
    .SUBSCRIBE-button:hover{
        opacity: 0.8;
    }
    .SUBSCRIBE-button:active{
    opacity: 0.5;
    }
    .JOIN-button{
        background-color: white;
        border-color: rgb(21, 102, 233);
        color: rgb(21, 102, 233);
        border-radius: 2px;
        border-style: solid;
        border-width: 1px;
        cursor: pointer;
        margin-right: 8px;
        transition: background-color 0.15s,color 0.15s;
        padding-left: 16px;
        padding-right: 16px;
        padding-bottom: 9px;
        padding-top: 9px;
    }
    .JOIN-button:hover{
        background-color:rgb(21, 102, 233) ;
        color: white;
    }
    .JOIN-button:active{
        opacity: 0.8;
    }
    .Tweet-button{
        background-color: rgb(24, 124, 237);
        color: white;
        border: none;
        border-radius: 18px;
        height: 36px;
        width: 74px;
        font-weight: bold;
        font-size: 15px;
        cursor: pointer;
        transition: box-shadow 0.15s;
    }
    .Tweet-button:hover{
        box-shadow:5px 5px 10px rgba(0,0,0,0.15) ;
    }

</style>
```
