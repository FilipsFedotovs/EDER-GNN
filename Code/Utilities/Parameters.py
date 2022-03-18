#This is the list of parameters that EDER-GNN uses for reconstruction, model training etc. There have been collated here in one place for the user convenience
# Part of EDER-GNN package
#Made by Filips Fedotovs
#Current version 1.0

######List of naming conventions
Hit_ID='ID'
x='x' #Column name x-coordinate of the track hit
y='y' #Column name for y-coordinate of the track hit
tx='TX' #Column name x-coordinate of the track hit
ty='TY' #Column name for y-coordinate of the track hit
z='z' #Column name for z-coordinate of the track hit
FEDRA_Track_ID='FEDRATrackID' #Column nameActual track id for FEDRA (or other reconstruction software)
FEDRA_Track_QUADRANT='quarter' #Quarter of the ECC where the track is reconstructed If not present in the data please put the Track ID (the same as above)
MC_Track_ID='MCTrack'  #Column name for Track ID for MC Truth reconstruction data
MC_Event_ID='MCEvent' #Column name for Event id for MC truth reconstruction data (If absent please enter the MCTrack as for above)


######List of geometrical constain parameters
stepX=2000
stepY=2000
stepZ=6000
cut_dt=0.2
cut_dr=60


# MaxSLG=4000
# MaxSTG=50#This parameter restricts the maximum length of of the longitudinal and transverse distance between track segments.
# MinHitsTrack=2
# MaxTrainSampleSize=50000
# MaxValSampleSize=100000
# MaxDoca=50
# MinAngle=0 #Seed Opening Angle (Magnitude) in radians
# MaxAngle=1 #Seed Opening Angle (Magnitude) in radians
#
#
#
# ##Model parameters
# pre_acceptance=0.5
# post_acceptance=0.5
# #pre_vx_acceptance=0.662
# resolution=50
# MaxX=2000.0
# MaxY=500.0
# MaxZ=20000.0
# Pre_CNN_Model_Name='1T_50_SHIP_PREFIT_1_model'
# Post_CNN_Model_Name='1T_50_SHIP_POSTFIT_1_model'
# ModelArchitecture=[[6, 4, 1, 2, 2, 2, 2], [], [],[], [], [1, 4, 2], [], [], [], [], [7, 1, 1, 4]]

