#ifndef MSDF_DEMO_H
#define MSDF_DEMO_H

#include <algorithm>

#include "MsdfFont.h"

// The MSDF demo scene shared by the windowed sample and the headless test so the
// two render identical content.
//
// The scene is authored in a fixed 1280x720 reference space and then uniformly
// scaled to fit (and centered within) whatever width x height surface it is
// given. Scaling the font sizes -- rather than bitmap-scaling a fixed-size
// render -- keeps the text crisp at any surface size, which is the whole point
// of MSDF. At exactly 1280x720 the scale is 1 and nothing moves.
inline void buildMsdfDemoScene(MsdfTextRenderer& renderer, MsdfFont* font, float width, float height)
{
    const float REF_W = 1280.0f;
    const float REF_H = 720.0f;
    const float s = std::min(width / REF_W, height / REF_H);
    const float ox = (width - REF_W * s) * 0.5f;  // center the reference canvas
    const float oy = (height - REF_H * s) * 0.5f;
    auto X = [&](float rx) { return ox + rx * s; };  // reference x -> surface x
    auto Y = [&](float ry) { return oy + ry * s; };  // reference y -> surface y
    auto Sz = [&](float rs) { return rs * s; };       // reference size -> surface size

    const float4 white{ 0.96f, 0.97f, 0.98f, 1.0f };
    const float4 grey{ 0.60f, 0.63f, 0.68f, 1.0f };
    const float4 red{ 0.94f, 0.33f, 0.31f, 1.0f };
    const float4 green{ 0.30f, 0.78f, 0.47f, 1.0f };
    const float4 blue{ 0.36f, 0.60f, 0.98f, 1.0f };
    const float4 yellow{ 0.98f, 0.80f, 0.25f, 1.0f };

    const float m = 64.0f; // reference-space left margin

    renderer.clear();

    renderer.addText(font, "NoGraphicsAPI MSDF text", X(m), Y(120.0f), Sz(88.0f), white);
    renderer.addText(font, "Inter, rendered from a multi-channel signed distance field.",
                     X(m), Y(174.0f), Sz(30.0f), grey);

    // Size ramp: the same line at increasing sizes, all crisp from one atlas.
    // y and line heights are accumulated in reference space, then mapped.
    float ry = 270.0f;
    for (float px : { 18.0f, 26.0f, 36.0f, 44.0f })
    {
        renderer.addText(font, "Sphinx of black quartz, judge my vow 0123456789",
                         X(m), Y(ry), Sz(px), blue);
        ry += font->lineHeight(px) + 14.0f;
    }

    // Color row: each word placed with measure() (the sizing helper a layout
    // library such as Clay would call), advancing in reference space.
    const float rowSize = 60.0f;
    const float rcy = 560.0f;
    float rcx = m;
    auto word = [&](const char* str, float4 color)
    {
        renderer.addText(font, str, X(rcx), Y(rcy), Sz(rowSize), color);
        rcx += font->measure(str, rowSize).x;
    };
    word("Red ", red);
    word("Green ", green);
    word("Blue ", blue);
    word("Yellow", yellow);

    // MTSDF outline: blue fill under a yellow outline, near the bottom.
    renderer.addText(font, "Outlined", X(m), Y(672.0f), Sz(84.0f), blue, 0.0f, yellow, Sz(3.0f));

    renderer.upload();
}

#endif // MSDF_DEMO_H
