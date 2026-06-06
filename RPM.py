class RPM(C2PSA):

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(RPMBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        (a, b) = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
