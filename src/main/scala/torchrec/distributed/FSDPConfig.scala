package torchrec.distributed

case class FSDPConfig(
  rank: Int = 0,
  worldSize: Int = 1,
  masterAddr: String = "localhost",
  masterPort: Int = 29500,
  shardingStrategy: String = "FULL_SHARD",
  bucketCapMB: Int = 25,
  backwardPrefetch: String = "BACKWARD_PRE",
  autoWrapPolicy: Option[String] = None,
  frozenParamDiscriminator: Option[String] = None
)

object FSDPConfigBuilder {
  def apply(): FSDPConfigBuilder = new FSDPConfigBuilder()

  class FSDPConfigBuilder {
    private var rank: Int = 0
    private var worldSize: Int = 1
    private var masterAddr: String = "localhost"
    private var masterPort: Int = 29500
    private var shardingStrategy: String = "FULL_SHARD"
    private var bucketCapMB: Int = 25
    private var backwardPrefetch: String = "BACKWARD_PRE"

    def rank(r: Int): this.type = { rank = r; this }
    def worldSize(ws: Int): this.type = { worldSize = ws; this }
    def masterAddr(addr: String): this.type = { masterAddr = addr; this }
    def masterPort(port: Int): this.type = { masterPort = port; this }
    def shardingStrategy(ss: String): this.type = { shardingStrategy = ss; this }
    def bucketCapMB(bc: Int): this.type = { bucketCapMB = bc; this }
    def backwardPrefetch(bp: String): this.type = { backwardPrefetch = bp; this }

    def build(): FSDPConfig = FSDPConfig(
      rank = rank,
      worldSize = worldSize,
      masterAddr = masterAddr,
      masterPort = masterPort,
      shardingStrategy = shardingStrategy,
      bucketCapMB = bucketCapMB,
      backwardPrefetch = backwardPrefetch
    )
  }
}
