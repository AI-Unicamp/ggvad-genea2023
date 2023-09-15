import os
from retarget import MotionRetargeting, SimpleMotionRetargeting
from surface import GetCalibrationFromBVHS
import time
import mathutils


# Generate Surface Calibration ###############################################
def GenerateSurfaceCalibration():
    start = time.time()
    GetCalibrationFromBVHS('..\\data\\surface\\calibration_takes\\rodolfo\\Frente001_mcp.bvh', '..\\data\\surface\\calibration_takes\\rodolfo\\Cabeca_mcp.bvh', '..\\data\\surface\\calibration_takes\\rodolfo\\Costas002_mcp.bvh', savefile=True, debugmode=True)
    print('Surface Calibration done. %s seconds.' % (time.time()-start))


def SurfaceMotionRetargeting(computeEgo=True, computeIK=True, adjustOrientation=True, saveFile=True, saveInitAndFull=True):
    targettpose = os.path.abspath('..\\data\\input\\tpose_talita.bvh')
    targetsurface = os.path.abspath('..\\data\\surface\\surface_talita.csv')
    skeletonmappath = None  # Use standard
    sourcesurface = os.path.abspath('..\\data\\surface\\surface_rodolfo.txt')

    sourceanimations = ['..\\data\\input\\Mao_na_frente_mcp.bvh']

    out_path = os.path.abspath('..\\data\\output')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for path in sourceanimations:
        starttime = time.time()
        MotionRetargeting.importance = []
        tgtAnim, tgtSurf, srcAnim, srcSurf, tgtAnim_onlyInitial, ego = MotionRetargeting(path, sourcesurface, targettpose, targetsurface, skeletonmappath,  computeEgo=computeEgo, computeIK=computeIK, adjustOrientation=adjustOrientation, saveFile=saveFile, saveInitAndFull=saveInitAndFull, out_path=out_path)
        print('Total time: %s seconds.' % (time.time()-starttime))
        mathutils.printLog()


def SkeletonMotionRetargeting(srcAnimPath, tgtSkelPath, outputPath, srcSkeletonMap=None, tgtSkeletonMap=None):
    tgtAnimation = SimpleMotionRetargeting(srcAnimPath,
                                           tgtSkelPath,
                                           outputPath,
                                           srcSkeletonMap,
                                           tgtSkeletonMap,
                                           forceFaceZ=True,
                                           trackProgress=True,
                                           frameStop=3000)


# GenerateSurfaceCalibration()
# SurfaceMotionRetargeting()


# SkeletonMotionRetargeting(srcAnimPath='..\\data\\input\\Mao_na_frente_mcp.bvh',
#                           tgtSkelPath=os.path.abspath('..\\data\\input\\tpose_aragor.bvh'),
#                           outputPath=os.path.abspath('..\\data\\output'),
#                           srcSkeletonMap=None,
#                           tgtSkeletonMap='..\\data\\surface\\skeletonmap_aragor.csv',
#                           )

SkeletonMotionRetargeting(srcAnimPath='F:\\Downloads\\Trinity\\FBXs\\q_NaturalTalking_001__x2d.bvh',
                          tgtSkelPath=os.path.abspath('..\\data\\input\\tpose_aragor.bvh'),
                          outputPath=os.path.abspath('..\\data\\output'),
                          srcSkeletonMap=None,
                          tgtSkeletonMap=os.path.abspath('..\\data\\surface\\skeletonmap_aragor.csv'),
                          )

# SkeletonMotionRetargeting(srcAnimPath='F:\\Downloads\\Trinity\\FBXs\\q_NaturalTalking_001__x2d.bvh',
#                           tgtSkelPath='D:\\deep-motion-editing-master\\style_transfer\\data\\xia_test\\angry_13_000.bvh',
#                           outputPath=os.path.abspath('..\\data\\output'),
#                           srcSkeletonMap=None,
#                           tgtSkeletonMap=os.path.abspath('..\\data\\surface\\skeletonmap_xia.csv'),
#                           )
print('Done')


# F:\Downloads\Trinity\FBXs
