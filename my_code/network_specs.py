# also called vgg A
vgg_11 = [
    (3,64,2), # 64 3x3 filters with 2x2 maxpooling after
    (3,128,2),
    (3,256,1), # no maxpool after
    (3,256,2),
    (3,512,1),
    (3,512,2),
    (3,512,1),
    (3,512,2),
    (1, 4096), # FC layer
    (1, 4096),
    (1, 5), # FC, probably thin this out a different way
    (1, 5) # softmax
]

# only difference is smaller pooling window
planktonnet_lesspool = [
    (3,32,1),
    (3,16,2),
    (3,64,1),
    (3,32,2),
    (3,128,1),
    (3,128,1),
    (3,64,2),
    (3,256,1),
    (3,256,1),
    (3,128,2),
    (1, 512),
    (1, 512),
    (1, 5),
    (1, 5)
]

planktonnet_lesspool_lite = [
    (3,16,2),
    (3,32,2),
    (3,64,1),
    (3,32,2),
    (3,128,1),
    (3,64,2),
    (1, 256),
    (1, 256),
    (1, 5),
    (1, 5)
]

galaxynet = [
    (6,32,2),
    (5,64,2),
    (3,128,1),
    (3,128,2),
    (1,2048),
    (1,2048),
    (1,10)
]

ciresan2012 = [
    (4, 20, 2),
    (5, 40, 3),
    (1, 150),
    (1, 10)
]

ciresan2012_cuco = [
    (4, 32, 2),
    (5, 64, 3),
    (1, 150),
    (1, 10)
]

ciresan2012_cuco_pad1 = [
    (3, 32, 2),
    (3, 64, 3),
    (1, 150),
    (1, 10)
]

ciresan2012_cuco_pad1_galaxyapprox = [
    (3, 32, 2),
    (3, 32, 2),
    (3, 64, 3),
    (1, 150),
    (1, 10)
] # TtL7LEANzxPpHamJkyC3wTeSCITV1moO

ciresan2012_cuco_pad1_galaxyapprox2 = [
    (3, 32, 2),
    (3, 32, 2),
    (3, 32, 2),
    (3, 64, 3),
    (1, 150),
    (1, 10)
] # LWjTQpwwJDsm8XQRjAi8Q0ZijcAuiawV

galaxynet_lessFC = [
    (6,32,2),
    (5,64,2),
    (3,128,1),
    (3,128,2),
    (1,150),
    (1,10)
] # rtCZoJXXKL3feLluk95A4bjXZy6zDsMf

ciresan2012_cuco_pad1_galaxyapprox3 = [
    (3, 32, 2),
    (3, 32, 2),
    (3, 64, 3),
    (1,2048),
    (1,2048),
    (1,10)
] # B5blhmPodIha4ZPxJ0NoWjirDpOVtTpA