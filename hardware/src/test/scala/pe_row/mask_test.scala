package pe_row

import org.scalatest._
import chiseltest._
import chisel3._
import chisel3.util._
import chisel3.experimental.FixedPoint

import scala.util.Random
import scala.math.exp
import scala.math.abs
import scala.math.max

class MaskTest extends FlatSpec with ChiselScalatestTester with Matchers {
  behavior of "Dense_PE_Array"
  it should "produce right output" in {
    test(new Dense_PE_Array(20, 8, 0, 11, 8, 8)) { c =>
      def relativeDiff(x: Double, y: Double) = abs(x - y) / max(x, y)

      // Prepare Data
      val r = new Random()

      val Q =
        for (i <- 0 until 8) yield for (j <- 0 until 8) yield r.nextInt(16)

      val K =
        for (i <- 0 until 8) yield for (j <- 0 until 8) yield r.nextInt(16)

      val S =
        for (i <- 0 until 8)
          yield for (j <- 0 until 8)
            yield exp(
              (for (k <- 0 until 8)
                yield Q(i)(k) * K(k)(j)).reduce(_ + _) / 256.0
            )

      val expSum = for (i <- 0 until 8) yield S(i).reduce(_ + _)

      // Clear
      c.io.c_astate.poke(c.a_clear)
      c.io.c_pestate.poke(c.pe_clear)
      c.clock.step(1)

      // Calc
      c.io.c_astate.poke(c.a_idle)
      c.io.c_pestate.poke(c.pe_calc)
      for (i <- 0 until 24) {
        for (j <- 0 until 8) {
          c.io.l_in(j).poke(if (j <= i && i - j < 8) Q(j)(i - j).U else 0.U)
          c.io.t_in(j).poke(if (j <= i && i - j < 8) K(i - j)(j).U else 0.U)
        }
        c.clock.step(1)
      }
      c.io.c_astate.poke(c.a_idle)
      c.io.c_pestate.poke(c.pe_move)
      c.clock.step(1)
      c.io.c_astate.poke(c.a_calc)
      val s = Array.ofDim[Double](8)
      for (i <- 0 until 8) {
        for (j <- 0 until 8) {
          assert(
            abs(
              relativeDiff(S(j)(7 - i), c.io.s_out(j).peek().litToDouble)
            ) < 5e-2
          )
          s(j) = s(j) + c.io.s_out(j).peek().litToDouble
        }
        c.clock.step(1)
      }
      // Check result
      c.io.c_astate.poke(c.a_idle)
      c.io.c_pestate.poke(c.pe_idle)
      for (j <- 0 until 8) {
        assert(
          relativeDiff(expSum(j), c.io.exp_sum(j).peek().litToDouble) < 5e-2
        )
      }
    }
  }
}
