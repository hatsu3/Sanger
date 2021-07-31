package pe_row

import org.scalatest._
import chiseltest._
import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

import scala.util.Random
import scala.math.{exp, abs, max, round}

class PeRowTest extends FlatSpec with ChiselScalatestTester with Matchers {
  behavior of "Sparse_PE_Array"
  it should "produce right output" in {
    def relativeDiff(x: Double, y: Double) = abs(x - y) / max(x, y)

    val bits = 32
    val point = 16
    test(
      new Sparse_PE_Array(
        4,
        bits,
        point,
        8,
        8,
        4,
        4,
        3
      )
    ) { c =>
      // Prepare Data
      val r = new Random()

      val Q =
        for (i <- 0 until 8)
          yield for (j <- 0 until 8) yield round(r.nextFloat() * 16) / 16.0

      val K =
        for (i <- 0 until 8)
          yield for (j <- 0 until 8) yield round(r.nextFloat() * 16) / 16.0

      val V =
        for (i <- 0 until 8)
          yield for (j <- 0 until 8) yield round(r.nextFloat() * 16) / 16.0

      val choices = for (i <- 0 until 8) yield i
      val select =
        for (i <- 0 until 8)
          yield r.shuffle(choices).slice(0, 4).sortWith(_ < _)

      val S =
        for (i <- 0 until 8)
          yield for (j <- 0 until 8)
            yield
              if (select(i).contains(j))
                exp(
                  (for (k <- 0 until 8)
                    yield Q(i)(k) * K(k)(j)).reduce(_ + _)
                )
              else 0.0
      val expSum = for (i <- 0 until 8) yield S(i).reduce(_ + _)

      val O =
        for (i <- 0 until 8)
          yield for (j <- 0 until 8)
            yield (for (k <- 0 until 8)
              yield S(i)(k) * V(k)(j)).reduce(_ + _)

      // Configure
      for (i <- 0 until 8)
        for (j <- 0 until 4) {
          c.io.c1(i)(j).poke(select(i)(j).U)
          c.io.c2(i)(j).poke(select(i)(j).U)
          if (j != 3)
            c.io.c4(i)(j).poke((select(i)(j + 1) - select(i)(j) - 1).U)
          else
            c.io.c4(i)(j).poke((7 - select(i)(j)).U)
        }

      // Clear
      c.io.clr.poke(1.B)
      c.io.c3.poke(c.acc_clear)
      c.io.c5.poke(c.exp_idle)
      c.clock.step(1)

      // Stage 1
      c.io.clr.poke(0.B)
      c.io.c3.poke(c.acc_self)
      for (i <- 0 until 24) {
        for (j <- 0 until 8) {
          c.io
            .left_in(j)
            .poke(
              FixedPoint.fromDouble(
                if (j <= i && i - j < 8) Q(j)(i - j) else 0,
                bits.W,
                point.BP
              )
            )

          c.io
            .top_in(j)
            .poke(
              FixedPoint
                .fromDouble(
                  if (j <= i && i - j < 8) K(i - j)(j) else 0,
                  bits.W,
                  point.BP
                )
            )
        }
        c.clock.step(1)
      }

      // Calc exp
      c.io.c3.poke(c.acc_idle)
      c.clock.step(1)
      c.io.c5.poke(c.exp_calc)
      c.clock.step(2)

      // Stage 2
      c.io.c3.poke(c.acc_left)
      c.io.c5.poke(c.exp_idle)
      for (i <- 0 until 8)
        for (j <- 0 until 4)
          c.io.c1(i)(j).poke(8.U)

      for (i <- 0 until 24) {
        for (j <- 0 until 8) {
          c.io
            .top_in(j)
            .poke(
              FixedPoint.fromDouble(
                if (j <= i && i - j < 8) V(j)(i - j) else 0,
                bits.W,
                point.BP
              )
            )
        }
        c.clock.step(1)

        if (i >= 8) {
          for (j <- 0 until 8)
            if (i - 8 - j >= 0 && i - 8 - j < 8) {
              assert(
                relativeDiff(
                  O(j)(i - 8 - j),
                  c.io.out(j).peek().litToDouble
                ) < 5e-2
              )
            }
        }
      }
      c.io.c3.poke(c.acc_idle)
      c.io.c5.poke(c.exp_move)

      c.clock.step(5)

      // Check sum of scores
      for (i <- 0 until 8) {
        assert(
          relativeDiff(expSum(i), c.io.score_sum(i).peek().litToDouble) < 5e-2
        )
      }
    }
  }
}
