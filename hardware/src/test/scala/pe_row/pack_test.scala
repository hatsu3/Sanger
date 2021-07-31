package pe_row

import org.scalatest._
import chiseltest._
import chisel3._

import scala.util.Random

class PackTest extends FlatSpec with ChiselScalatestTester with Matchers {
  behavior of "Pack"
  it should "produce right output" in {
    test(new Pack(8, 4, 4, 3, 3, 8)) { c =>
      // Prepare Data
      val r = new Random()

      val choices = for (i <- 0 until 8) yield i
      val select =
        for (i <- 0 until 8)
          yield r.shuffle(choices).slice(0, 4).sortWith(_ < _)

      val binaries =
        for (i <- 0 until 8)
          yield for (j <- 0 until 8)
            yield if (select(i).contains(j)) 1 else 0

      // Check
      c.io.c_in.poke(c.in_clear)
      c.clock.step(1)
      c.io.c_in.poke(c.in_read)
      for (i <- 0 until 8) {
        var shouldWrite = false
        for (j <- 0 until 8) {
          c.io.binaries(j).poke(binaries(j)(i).B)
          shouldWrite = shouldWrite || binaries(j)(i) == 1
        }
        c.clock.step(1)

        c.io.maskWrite.expect(shouldWrite.B)
        for (j <- 0 until 8) {
          if (binaries(j)(i) == 1)
            c.io.mask(j).expect((i).U)
          else
            c.io.mask(j).expect(8.U)
        }
      }
      for (j <- 0 until 8)
        c.io.binaries(j).poke(1.B)
      c.io.emit.expect(1.B)
    }
  }
}
