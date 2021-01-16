def threeD_plot(num_of_seed,image,seed,patch,simplified_points,batch,size_for_image=1,size_for_seed=1,size_for_patch=1,alpha_for_image=0,show_inter=False):
    import matplotlib
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import time
    collor=["#ff0000","#b5b500","#00ff00","#0000ff","#ff00d4","#00eaff","#00ff9d","#ffaa00","#ffff00",]
    matplotlib.use('WebAgg')
    ax2 = plt.axes(projection="3d")
    if torch.is_tensor(image):
        arr=image[batch]
        x=arr.shape[0]
        if(x!=3): arr=arr.permute(1,0)
        imgs_cpu=arr.cpu()
        imgs = imgs_cpu.numpy()
        ax2.scatter3D(imgs[0],imgs[1],imgs[2],c='b',alpha=alpha_for_image,s=size_for_image)
    if torch.is_tensor(seed):
        arr=seed[batch]
        x=arr.shape[0]
        if(x==3): arr=arr.permute(1,0) 
        for i in range(num_of_seed):
            imgs_cpu=arr[i].cpu()
            imgs = imgs_cpu.detach().numpy()
            ax2.scatter3D(imgs[0],imgs[1],imgs[2],c='k',s=size_for_seed)
    if torch.is_tensor(simplified_points):
        arr=simplified_points[batch]
        x=arr.shape[0]
        if(x!=3): arr=arr.permute(1,0) 
        imgs_cpu=arr.cpu()
        imgs = imgs_cpu.detach().numpy()
        ax2.scatter3D(imgs[0],imgs[1],imgs[2],c='k',s=size_for_seed)
    if torch.is_tensor(patch):
        arr=patch[batch]
        x=arr.shape[0]
        y=arr.shape[1]
        z=arr.shape[2]
        if(z==num_of_seed): 
            arr=arr.permute(2,1,0)
            temp =x
            x=z
            z=temp
        if(y==num_of_seed):
            arr=arr.permute(1,0,2)
            temp =x
            x=y
            y=temp
        if(y!=3): arr=arr.permute(0,2,1)
        if(x!=num_of_seed): arr=arr.permute(1,0)
        for i in range(num_of_seed):
            imgs_cpu=arr[i].cpu()
            imgs = imgs_cpu.detach().numpy()
            ax2.scatter3D(imgs[0],imgs[1],imgs[2],c=collor[i],s=size_for_patch)
    plt.savefig(time.strftime("%Y%m%d-%H%M%S"+str(batch)))
    if show_inter:
        plt.show()




    