package torchrec.distributed

enum ShardingStrategy {
  case FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
}
