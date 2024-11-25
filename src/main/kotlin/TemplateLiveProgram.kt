import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.extra.noclear.NoClear
import org.openrndr.extra.noise.random
import org.openrndr.extra.noise.uniform
import org.openrndr.extra.noise.uniformRing
import org.openrndr.extra.olive.oliveProgram
import org.openrndr.extra.quadtree.Quadtree
import org.openrndr.shape.Rectangle
import org.openrndr.shape.clamp
import ddf.minim.Minim
import ddf.minim.analysis.FFT
import ddf.minim.analysis.HannWindow
import org.openrndr.Fullscreen
import org.openrndr.KEY_ESCAPE
import org.openrndr.animatable.Animatable
import org.openrndr.animatable.easing.Easing
import org.openrndr.collections.push
import org.openrndr.draw.*
import org.openrndr.extra.color.spaces.ColorOKHSLa
import org.openrndr.extra.compositor.*
import org.openrndr.extra.fx.blur.BoxBlur
import org.openrndr.extra.fx.color.ColorCorrection
import org.openrndr.extra.fx.color.ColorMix
import org.openrndr.extra.gui.GUI
import org.openrndr.extra.minim.minim
import org.openrndr.extra.parameters.BooleanParameter
import org.openrndr.extra.parameters.DoubleParameter
import org.openrndr.extra.parameters.IntParameter
import org.openrndr.math.*
import org.openrndr.writer
import kotlin.math.*

/**
 *  This is a template for a live program.
 *
 *  It uses oliveProgram {} instead of program {}. All code inside the
 *  oliveProgram {} can be changed while the program is running.
 */

fun main() = application {
    configure {
        width = 1500
        height = 1000
        windowResizable = true
    }

    oliveProgram {
        val minim = minim()
        if (minim.lineOut == null) {
            println("Err: minim line out not initialized")
            application.exit()
        }

        val lineIn = minim.getLineIn(Minim.MONO, 2048, 48000f)
        if (lineIn == null) {
            println("Err: minim line in not initialized")
            application.exit()
        }
        val fft = FFT(lineIn.bufferSize(), lineIn.sampleRate())
        fft.window(HannWindow())
        fft.linAverages(10)

        val nightAnimation = object : Animatable() {
            var bgNight = 0.0
            var textAnim = 0.0
            var textShuffle = 0.0
            var textVisible = 0.0
        }

        val SEPARATION = 0.7
        val WALL_SEPARATION = 9.0
        val ALIGNMENT = 0.15
        val COHESION = 0.03

        val bounds = Rectangle(0.0, 0.0, width.toDouble(), height.toDouble())

        val qt = Quadtree<Agent>(bounds) {
            it.position
        }

        val agents = List(1150) {
            val a = Agent(
                Vector2.uniform(Vector2.ZERO, Vector2(width.toDouble(), height.toDouble())),
                Vector2.uniformRing(innerRadius = 1.0, outerRadius = 1.0),
                id = it
            )

            a
        }.toMutableList()

        extend(NoClear())

        class Mask : Filter1to1(filterShaderFromCode("""
/* based on "Brightness, contrast, saturation" by WojtaZam: https://www.shadertoy.com/view/XdcXzn */
const vec2 saturation_dn = vec2(-0.1, -0.1);
uniform float contrast_d;
uniform float contrast_n;
uniform float brightness_d;
uniform float brightness_n;
uniform float hueShift_d;
uniform float hueShift_n;

uniform sampler2D mask;

uniform sampler2D tex0;
in vec2 v_texCoord0;
out vec4 o_color;

mat4 brightnessMatrix(float brightness) {
    return mat4(1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    brightness, brightness, brightness, 1);
}

mat4 contrastMatrix(float contrast) {
    float t = (1.0 - contrast) / 2.0;
    return mat4(contrast, 0, 0, 0,
    0, contrast, 0, 0,
    0, 0, contrast, 0,
    t, t, t, 1 );
}

mat4 saturationMatrix(float saturation) {
    vec3 luminance = vec3(0.3086, 0.6094, 0.0820);
    float oneMinusSat = 1.0 - saturation;
    vec3 red = vec3(luminance.x * oneMinusSat);
    red += vec3(saturation, 0, 0);

    vec3 green = vec3(luminance.y * oneMinusSat);
    green += vec3(0, saturation, 0);

    vec3 blue = vec3(luminance.z * oneMinusSat);
    blue += vec3(0, 0, saturation);

    return mat4(red, 0,
        green, 0,
        blue, 0,
        0, 0, 0, 1 );
}

// from starea's https://www.shadertoy.com/view/MdjBRy, which in turn remixed it from mAlk's https://www.shadertoy.com/view/MsjXRt
vec3 shiftHue(in vec3 col, in float Shift) {
    vec3 P = vec3(0.55735) * dot(vec3(0.55735), col);
    vec3 U = col - P;
    vec3 V = cross(vec3(0.55735), U);
    col = U * cos(Shift * 6.2832) + V * sin(Shift * 6.2832) + P;
    return col;
}

void main() {
    vec4 color = texture(tex0, v_texCoord0);
    vec4 mask = texture(mask, v_texCoord0);
    
    vec3 inverted = 1.0 - color.rgb;
    color.rgb = mix(inverted, color.rgb, mask.r);
    
    float brightness = mix(brightness_d, brightness_n, mask.r);
    float saturation = mix(saturation_dn.x, saturation_dn.y, mask.r);
    float contrast = mix(contrast_d, contrast_n, mask.r);
    float hueShift = mix(hueShift_d, hueShift_n, mask.r);
    float gamma = mask.g;
    
    vec4 nc = (color.a == 0.0) ? vec4(0.0) : vec4(color.rgb / color.a, color.a);
    nc.rgb = pow(nc.rgb, vec3(gamma));
    nc.rgb = shiftHue(nc.rgb, (hueShift/360.0));
    vec4 cc = brightnessMatrix(brightness) * contrastMatrix((contrast + 1.0)) * saturationMatrix(saturation + 1.0) * nc;
    
    o_color = vec4(cc.rgb, 1.0) * color.a;
}
        """,
        "dayNight"))

        val fadeShader = """
        #version 330
        precision highp float;
        
        // -- part of the filter interface, every filter has these
        in vec2 v_texCoord0;
        uniform sampler2D tex0;
        uniform sampler2D tex1;
        uniform sampler2D fadeMask;
        uniform float fade_d;
        uniform float fade_n;
        out vec4 o_color;

        void main() {
            vec4 color0 = texture(tex0, v_texCoord0);
            vec4 color1 = texture(tex1, v_texCoord0);
            float fade = mix(fade_d, fade_n, texture(fadeMask, v_texCoord0).r);
            color0 += color1;
            color0.rgb *= fade;
            o_color = color0;
        }
        """
        class Fade : Filter2to1(filterShaderFromCode(fadeShader, "fade-shader"))
        val fade = Fade()
        val blur = BoxBlur()
        blur.window = 1

        val gui = GUI()
        val settings = object {
            @IntParameter("Day cycle", 10, 360)
            var daycycle: Int = 120

            @BooleanParameter("night")
            var night: Boolean = false

            @BooleanParameter("day")
            var day: Boolean = false

            @BooleanParameter("text")
            var text: Boolean = false

            @DoubleParameter("hueShift_d", -180.0, 180.0)
            var hueShift_d: Double = 20.0
            @DoubleParameter("hueShift_n", -180.0, 180.0)
            var hueShift_n: Double = 0.0

            @DoubleParameter("contrast_d", 0.0, 1.0)
            var contrast_d: Double = 0.0
            @DoubleParameter("contrast_n", 0.0, 1.0)
            var contrast_n: Double = 0.0

            @DoubleParameter("brightness_d", -1.0, 1.0)
            var brightness_d: Double = 0.0
            @DoubleParameter("brightness_n", -1.0, 1.0)
            var brightness_n: Double = 0.0

            @DoubleParameter("fade_d", 0.99, 1.0)
            var fade_d: Double = 0.999
            @DoubleParameter("fade_n", 0.99, 1.0)
            var fade_n: Double = 0.995

            @DoubleParameter("gammaBg_d", 0.1, 10.0)
            var gammaBg_d: Double = 3.0
            @DoubleParameter("gammaBg_n", 0.1, 10.0)
            var gammaBg_n: Double = 0.9
            @DoubleParameter("gammaTxt_d", 0.1, 10.0)
            var gammaTxt_d: Double = 8.0
            @DoubleParameter("gammaTxt_n", 0.1, 10.0)
            var gammaTxt_n: Double = 0.55
        }

        // -- this is why we wanted to keep a reference to gui
        gui.add(settings, "Settings")
        extend(gui)

        fun getColor(idx: Int, signal: Double, hueBaseline: Double = 0.0): ColorRGBa {
            val hue = -idx.toDouble() / agents.size * 170.0 + 30.0 + hueBaseline

            val sat = clamp((sin(idx / 10.0) * 0.5 + 0.5) * 0.5 + 0.5, 0.2, 0.95)

            val lightScale = mix(0.5, 1.0, nightAnimation.bgNight)
            val lig = clamp(signal * 0.08, 0.02, 0.65) * lightScale

            return ColorOKHSLa(
                hue,
                sat,
                lig,
                0.5
            ).toRGBa()
        }

        val composite = compose {
            val accum = colorBuffer(width, height, type = ColorType.FLOAT32)
            // val temp = colorBuffer(width, height, type = ColorType.FLOAT32)

            val frame = aside(colorType = ColorType.FLOAT32) {
                draw {
                    drawer.clear(ColorRGBa.BLACK)
                    drawer.strokeWeight = 0.0
                    drawer.stroke = ColorRGBa.BLACK

                    drawer.circles {
                        agents.forEachIndexed { idx, agent ->
                            val signal = agent.getSignal()

                            fill = getColor(idx, signal)

                            val size = signal * (0.1 + 0.04 * nightAnimation.bgNight) + 0.5 + nightAnimation.bgNight * 0.2
                            circle(agent.position, size)
                        }
                    }
                }
            }

            val mask = aside(colorType = ColorType.FLOAT32) {
                val font = loadFont("data/fonts/Valorax-lg25V.otf", 170.0)
                val fontSm = loadFont("data/fonts/Valorax-lg25V.otf", 50.0)

                val messages = arrayOf(
                    "Tanssijat",
                    "Dancers",
                    "Boids",
                    "Flock",
                    "Saltatores",
                    "Colorful",
                    "Varia eiu",
                    "OpenRNDR",
                    "Evoluutio",
                    "Emergenssi",
                    "Parvi",
                )
                draw {
                    val dayBg = Vector3(0.0, settings.gammaBg_d, 0.0)
                    val nightBg = Vector3(1.0, settings.gammaBg_n, 0.0)
                    val dayFill = Vector3(1.0, settings.gammaTxt_d, 0.0).mix(nightBg, nightAnimation.textAnim)
                    val nightFill = Vector3(0.0, settings.gammaTxt_n, 0.0).mix(dayBg, nightAnimation.textAnim)
                    val clear = dayBg.mix(nightBg, nightAnimation.bgNight)
                    val fill = dayFill.mix(nightFill, nightAnimation.bgNight)
                    drawer.clear(ColorRGBa.fromVector(clear))
                    drawer.fill = ColorRGBa.fromVector(fill)

                    if (nightAnimation.textVisible == 1.0 || settings.text) {



                        drawer.isolated {
                            writer {
                                val messageSm = "mix up!"
                                drawer.fontMap = fontSm
                                val w = textWidth(messageSm)

                                cursor = Cursor(0.0, height - 100.0)

                                val centering = Vector2(width - w, 0.0) * (sin(seconds * 0.1) * 0.5 + 0.5)
                                drawer.translate(centering)

                                text(messageSm)
                            }
                        }

                        drawer.fontMap = font

                        val messageIdx = if (nightAnimation.textShuffle > 0.0) {
                            val base = seconds
                            val idx = nightAnimation.textShuffle * messages.size * 2
                            ((base + idx) % messages.size).toInt()
                        } else { 0 }

                        val message = messages[messageIdx]

                        writer {
                            val w = textWidth(message)

                            cursor = Cursor(-w / 2, 0.0)

                            //drawer.translate(-w / 2, 0.0)
                            drawer.translate(width/2.0, height/2.0)
                            val centering = Vector2(8000.0, height/2.0 + 1000.0) * nightAnimation.textAnim
                            drawer.translate(centering)
                            drawer.scale(1.0 + nightAnimation.textAnim * 100.0)


                            text(message)
                        }


                    }
                }
            }

            layer {


                draw {
                    if (frameCount % (1 + 3*(1 - nightAnimation.bgNight.toInt())) == 0) {
                        blur.apply(accum, accum)
                    }
                    fade.parameters["fadeMask"] = mask.result
                    fade.parameters["fade_d"] = settings.fade_d
                    fade.parameters["fade_n"] = settings.fade_n
                    fade.apply(frame.result, accum, accum)
                    drawer.image(accum)
                }

                post(Mask()) {
                    parameters["mask"] = mask.result
                    parameters["hueShift_d"] = settings.hueShift_d
                    parameters["hueShift_n"] = settings.hueShift_n
                    parameters["contrast_d"] = settings.contrast_d
                    parameters["contrast_n"] = settings.contrast_n
                    parameters["brightness_d"] = settings.brightness_d
                    parameters["brightness_n"] = settings.brightness_n
                }
            }
        }

        extend {
            fft.forward(lineIn.mix)

            val toggle = if (nightAnimation.bgNight == 0.0) {
                if (settings.night) {
                    settings.night = false
                    true
                } else {
                    false
                }
            } else {
                if (settings.day) {
                    settings.day = false
                    true
                } else {
                    false//
                }
            }

            if (seconds % settings.daycycle < 0.1 && !nightAnimation.hasAnimations() && seconds > 1 || toggle) {
                nightAnimation.apply {
                    val nextVal = if (bgNight == 0.0) { 1.0 } else { 0.0 }

                    ::textVisible.animate(1.0, 0, Easing.None)
                    ::textShuffle.animate(1.0, 1200, Easing.CubicInOut, predelayInMs = 1600)
                    ::textShuffle.animate(0.0, 1200, Easing.CubicInOut, predelayInMs = 3600)
                    ::textAnim.animate(1.0, 2500, Easing.CubicInOut, predelayInMs = 6000)
                    ::textAnim.complete()
                    ::bgNight.animate(nextVal, 0, Easing.None)
                    ::textVisible.animate(0.0, 0, Easing.None)
                    ::textAnim.animate(0.0, 0, Easing.None)
                }
            }
            nightAnimation.updateAnimation()

            if (nightAnimation.textVisible == 1.0) {
                // Swap to randomize
                repeat(20) {
                    val idx0 = random(0.0, agents.size - 1.0).toInt()
                    val idx1 = random(0.0, agents.size - 1.0).toInt()
                    val temp = agents[idx0].position.copy()
                    agents[idx0].position = agents[idx1].position
                    agents[idx1].position = temp
                }
            }

            // Init quadtree
            qt.clear()
            agents.forEach { qt.insert(it) }

            agents.forEachIndexed { idx, agent ->
                val bandIdx = fft.freqToIndex(idx.toFloat() * 10)
                val amp = log10(1.0 + fft.getBand(bandIdx).toDouble()) * ln(idx.toDouble() + 1.0)
                agent.updateSignal(amp)
            }

            agents.forEachIndexed { idx, agent ->

                val query = qt.nearest(agent, 300.0)

                val (edgePos, distToEdge) = bounds.nearestPointInside(agent.position)
                var toNearestEdge = agent.position - edgePos
                toNearestEdge /= distToEdge * distToEdge
                agent.direction += toNearestEdge * WALL_SEPARATION

                if (query != null) {
                    val sim0 = agent.similarity(query.nearest, 100.0)
                    val diff = (1.0-sim0)

                    var toNearest = agent.position - query.nearest.position
                    val distToNearest = toNearest.length
                    toNearest /= distToNearest * distToNearest

                    agent.direction += toNearest * SEPARATION * (1.0 + diff)

                    var headingWeight = 0.001
                    var avgHeading = Vector2.ZERO
                    var positionWeight = 0.0;
                    var avgPosition = Vector2.ZERO


                    query.neighbours.forEach {
                        val similarity = it.similarity(agent, 40.0)
                        val weight = 2.0 * similarity - 1.0

                        headingWeight += similarity
                        avgHeading += it.direction * weight

                        positionWeight += weight
                        avgPosition += it.position * weight
                    }

                    avgHeading /= headingWeight
                    avgPosition /= positionWeight

                    val toAvgPosition = (agent.position - avgPosition).normalized

                    agent.direction = agent.direction.mix(avgHeading, ALIGNMENT)

                    agent.direction = agent.direction.mix(toAvgPosition, COHESION)
                }



                agent.direction = agent.direction.normalized


                agent.position += agent.direction * (agent.avgSignal50 * 20.0 + 1.0)

                agent.position.clamp(bounds)
            }

            composite.draw(drawer)
            /*drawer.clear(ColorRGBa.BLACK)
            drawer.fill = ColorRGBa.RED
            drawer.rectangles {
                val n = agents.size
                val w = 1.0 / n * width
                var minH = Double.MAX_VALUE
                var maxH = Double.MIN_VALUE
                repeat(n) {
                    val a = agents[it]
                    val signal = a.getSignal()
                    val x = w * it
                    val h = signal * 100.0
                    minH = if (h < minH) { h } else { minH }
                    maxH = if (h > maxH) { h } else { maxH }

                    val col = getColor(it, signal)
                    fill = col
                    stroke = col
                    rectangle(x, height - h, w, h)
                }

                // println(minH)
                // println(maxH)
                // println()
            }*/
        }

        keyboard.keyDown.listen {
            println(it.name)

            if (it.name == "g") {
                gui.visible = !gui.visible
            }
        }

    }
}

class Agent(var position: Vector2, var direction: Vector2, var signalDelta: Double = 0.0, val id: Int) {
    var signalQueue10 = ArrayDeque<Double>(10)
    var signalQueue50 = ArrayDeque<Double>(50)
    var totalSignal10 = 0.0
    var avgSignal10 = 0.0
    var totalSignal50 = 0.0
    var avgSignal50 = 0.0
    var lastSignal = 0.0

    fun updateSignal(newSignal: Double) {
        lastSignal = newSignal

        totalSignal10 += newSignal
        totalSignal50 += newSignal
        if (signalQueue10.size >= 10) {
            totalSignal10 -= signalQueue10.removeFirst()
        }
        if (signalQueue50.size >= 50) {
            totalSignal50 -= signalQueue50.removeFirst()
        }

        signalQueue10.push(newSignal)
        signalQueue50.push(newSignal)

        avgSignal10 = totalSignal10 / signalQueue10.size
        avgSignal50 = totalSignal50 / signalQueue50.size
    }

    fun getSignal(): Double {
        // return max(avgSignal10 / (avgSignal50 / 2) - 1.0, 0.0)
        return lastSignal / (avgSignal50 / 2 + 0.001)
    }

    fun similarity(other: Agent, tolerance: Double = 10.0): Double {
        val x = (other.id - this.id) / tolerance
        val exponent = -(x * x)
        return exp(exponent)
    }
}


fun Rectangle.nearestPointInside(p: Vector2): Pair<Vector2, Double> {
    val x0 = this.x
    val x1 = x0 + this.width
    val y0 = this.y
    val y1 = y0 + this.height

    val dx0 = p.x - x0;
    val dx1 = x1 - p.x;
    val dy0 = p.y - y0;
    val dy1 = y1 - p.y;
    val dx = if (dx0 < dx1) { dx0 } else { dx1 }
    val dy = if (dy0 < dy1) { dy0 } else { dy1 }

    val dist: Double
    val pos = if (dx < dy) {
        dist = dx
        if (dx0 < dx1) {
            Vector2(x0, p.y)
        } else {
            Vector2(x1, p.y)
        }
    } else {
        dist = dy
        if (dy0 < dy1) {
            Vector2(p.x, y0)
        } else {
            Vector2(p.x, y1)
        }
    }

    return Pair(pos, dist)
}