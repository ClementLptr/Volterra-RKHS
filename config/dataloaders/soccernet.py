import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/usr/users/volterrakernel/lepretre_cle/volterra/data/soccernet_labelized")
mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])